import subprocess

def compress_pdf(input_path, output_path=None, quality="ebook"):
    """
    Compress a pdf file specified by `input_path`.    
    quality:
        screen  : 低画質・超軽量
        ebook   : 中画質（おすすめ）
        printer : 高画質
        prepress: 最高画質
    """
    if output_path is None:
        output_path = input_path[:-4] + '_compressed.pdf'
    try:
        subprocess.run([
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS=/{quality}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            f"-sOutputFile={output_path}",
            input_path
        ])
    except:
        print("Install Ghostscript.\n Mac: `brew install ghostscript`\nUbuntu:`sudo apt install ghostscript`")
