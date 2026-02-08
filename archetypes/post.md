---
title: '{{ replace .File.ContentBaseName "-" " " | title }}'
date: "{{ .Date }}"
# weight: 1
# aliases: ["/first"]
tags: ["first"]
author: "Arav Tewari"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Description text"
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>"   # image path/url
    alt: "<alt text>"           # alt text
    caption: "<text>"           # display caption under cover
    relative: false             # when using page bundles set this to true
    hidden: true                # only hide on current single page
---