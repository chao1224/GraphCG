import gdown


if __name__ == "__main__":
    url = 'https://drive.google.com/uc?id=' + "11cjayJE36XHXnk6HqMXAu_U8Rza6FCch"
    output = "HierVAE.zip"
    gdown.download(url, output)
