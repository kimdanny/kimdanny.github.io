from typing import Tuple
import torch


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


# Adapted from https://github.com/facebookresearch/llama/blob/main/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # print(freqs.shape) # torch.Size([2])
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # print(t.shape) # torch.Size([20])
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # print(freqs.shape)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # print(freqs_cis.shape)
    return freqs_cis


# Adapted from https://github.com/facebookresearch/llama/blob/main/llama/model.py
def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device

    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf.
    # You may also benefit from https://blog.eleuther.ai/rotary-embeddings/.

    # reshape xq and xk to match the complex representation
    # original code
    # query_real, query_imag = (
    #     query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    # )
    # key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    xq_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2)).to(
        device=device
    )
    xk_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2)).to(
        device=device
    )

    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    freqs_cis = precompute_freqs_cis(head_dim, seqlen, theta=theta).to(device=device)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_).to(device=device)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # query_out = None
    # key_out = None
    # Return the rotary position embeddings for the query and key tensors
    # return query_out, key_out
    return xq_out.type_as(query), xk_out.type_as(key)
