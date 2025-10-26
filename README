# OH！This lazy person has finally taken action！
--- 

# hugo+Github

## 本地安装huge

参考 https://gohugo.com.cn/installation/

安装extend版本，输入以下命令查看版本号

```shell
hugo version
```


## 创建本地博客

```shell
hugo new site my-blog
cd my-blog
git init
git submodule add https://github.com/Daucloud/hugo-theme-void themes/void
echo 'theme = "void"' >> hugo.toml

```
hugo.toml 文件如下
```shell
baseURL = 'https://yourusername.github.io/'
languageCode = 'en-US'
title = 'Your Site Title'
theme = "void"

# Social media links
[params]
  [params.social]
    github = "https://github.com/yourusername"
    twitter = "https://twitter.com/yourusername"
    email = "your.email@example.com"
  [params.avatar]
    url = "https://example.com/your-avatar.jpg"

# Main menu
[[menus.main]]
name = 'Home'
pageRef = '/'
weight = 10

[[menus.main]]
name = 'Posts'
pageRef = '/posts'
weight = 20

[[menus.main]]
name = 'Tags'
pageRef = '/tags'
weight = 30

[[menus.main]]
name = 'About'
pageRef = '/about'
weight = 40
```

## 本地预览
- 本地依赖安装TailwindCSS，在my-blog目录下执行，得到package.json文件
```
npm init -y
npm install tailwindcss postcss autoprefixer
```
- 之后执行，如下命令
```shell
hugo server -D
```

## 托管到github

在`my-blog`目录下，创建目录结构：`.github/workflows/`，并在 workflows 目录中新建文件`deploy.yml`,`deploy.yml`如下

```shell
name: Deploy Hugo Site to Pages

on:
  push:
    branches: ["master"]   # 代码推送至 main 分支时触发
  workflow_dispatch:     # 支持手动触发

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          submodules: recursive  # 自动拉取主题子模块
          fetch-depth: 0

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: 'latest'
          # extended: true       # 若主题需扩展版，取消注释

      - name: Install dependencies
        run: hugo ci

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./public          # 上传生成的站点文件

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

绑定github repo
```shell
# 关联远程仓库（地址替换为刚复制的地址）
git remote add origin https://github.com/<你的用户名>/<你的用户名>.github.io.git
# 添加所有文件到暂存区并提交
git add .
git commit -m "Initial commit with Hugo site and Stack theme"
# 将本地 master 分支推送到 GitHub 并设为上游分支
git branch -M master
git push -u origin master
```

## 本地blog 撰写

```shell
#两种方式
#第一种：命令创建
hugo new content post/文章名/index.md
#第二种：手动添加
将写好的markdown文档添加到content/post里面

```
之后推送到github repo就好

# Ref

https://leonincs.github.io/p/hugo--github%E5%BF%AB%E9%80%9F%E6%90%AD%E5%BB%BA%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2/

https://github.com/Daucloud/hugo-theme-void/tree/391731556bb968bb41be59d4953f908f8f88b572