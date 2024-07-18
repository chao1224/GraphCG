import gdown

if __name__ == "__main__":
    # https://github.com/stevenygd/PointFlow/blob/master/README.md?plain=1#L48
    url = 'https://drive.google.com/uc?id=1sw9gdk_igiyyt7MqALyxZhRrtPvAn0sX'
    output = "ShapeNetCore.v2.PC15k.zip"
    gdown.download(url, output)
