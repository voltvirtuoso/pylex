
## P&ID OCR research update

The uploaded Sujawal screenshot and attached Halini P&ID expose known technical-drawing OCR failure modes: mixed horizontal/vertical text, small labels inside symbols, line-art touching characters, large variation in text scale, and patch boundary effects.

The Microsoft Engineering Document P&ID Digitization article reports that P&ID text detection is affected by poor resolution and crowded symbols. Its described preprocessing experiments include grayscale and binarization, while its tiling experiment found only marginal accuracy improvement and added latency for that particular workflow. The article uses Azure AI Document Intelligence for high-resolution OCR, which is not part of this Paddle-only project.

The Schlagenhauf, Netzer, and Hillinger paper on technical-drawing digitization separates text detection from recognition: it uses a dedicated detector to produce text boxes, then a recognizer trained on synthetic technical-drawing character sequences. It explicitly reports weaknesses of out-of-the-box OCR on vertical text near other drawing elements and argues that domain-specific synthetic training data is needed for reliable technical-drawing recognition.

The Stürmer, Graumann, and Koch PID2Graph paper describes patching large P&IDs and stitching patch results. Its preprocessing resizes a diagram to roughly 4500x7000 and uses patches with at least 50% overlap. This supports using local-resolution tiles and careful overlap reconciliation, but it is a broader digitization system rather than a drop-in OCR backend.

Official PaddleX/PaddleOCR text-line orientation documentation says the classifier has two classes, 0 degrees and 180 degrees. It is not a general 90-degree vertical-text classifier. This explains why an explicit targeted 90-degree retry is needed for vertical engineering labels.

Sources:
- https://devblogs.microsoft.com/ise/engineering-document-pid-digitization/
- https://arxiv.org/pdf/2205.02659
- https://arxiv.org/html/2411.13929v1
- https://paddlepaddle.github.io/PaddleX/3.4/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html
- https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/textline_orientation_classification.html
