---
title: "How to browse hidden folders/files on your Mac Finder"
categories:
  - random
tags:
  - Mac
toc: false
---
In your terminal by typing `ls -la` you can look at the hidden folders and files, 
but you can also set your Finder to show those hidden ones.  

In your terminal, type
```
defaults write com.apple.finder AppleShowAllFiles YES
```

Then to refresh your Finder, type
```
Killall Finder
```

To revert this setting, type NO instead of YES:
```
defaults write com.apple.finder AppleShowAllFiles NO
```
