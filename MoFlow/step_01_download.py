import gdown

if __name__ == "__main__":
    url = 'https://drive.google.com/uc?id=' + "1OTjo0R6Ps_jg9farWhBIk76zF2MCAR0D"
    output = "MoFlow.zip"
    gdown.download(url, output)
