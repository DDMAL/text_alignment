# cantus-alignment

Given an image of the text layer of a chant manuscript, and a transcript of text that lies entirely within that manuscript, finds the position of each syllable of that text on the page.

### Installation

Python 2.7+ only.
Requires [Gamera](https://gamera.informatik.hsnr.de/) to be installed in the same environment.
Requires [OCRopus](https://github.com/tmbdev/ocropy) as well. OCRopus is not an importable module but a set of command-line tools. Follow the installation instructions on their github. The important thing, whether installed globally or to a virtual environment, is that typing ```ocropy-rpred -h``` in a shell opened to the top-level folder of this project produces a help message.

OCRopus only supports Mac officially. I've gotten this project to work on Windows on my machine, since it only needs one component of OCRopus, but YMMV. On Windows, parallel processing doesn't work, and you might see a bunch of warnings pop up during the text recognition part.

### How To Run Locally
Run from alignToOCR.py. Just does one file at a time, at the moment - edit the parameters at the top of the ```__main__``` method to change what is processed. Given ```filename``` to process, this project will look for files at ```./png/filename_text.png``` and ```./png/filename_transcript.txt``` and output an image file with the results of the alignment visualized.
