# Reduce PDF File Size

There are multiple ways to reduce the size of a given PDF file. The easiest
one is to remove content (e.g., images) or pages.

```{testsetup}
pypdf_test_setup("user/file-size", {
    "example.pdf": "../resources/example.pdf",
})
```

## Removing duplication

Some PDF documents contain the same object multiple times. For example, if an
image appears three times in a PDF, it could be embedded three times. Or it can
be embedded once and referenced twice.

When adding data to a PdfWriter, the data is copied while respecting the original format.
For example, if two pages include the same image which is duplicated in the source document, the object will be duplicated in the PdfWriter object.

Additionally, when you delete objects in a document, pypdf cannot easily identify whether the objects are used elsewhere or not or if the user wants to keep them in. When writing the PDF file, these objects will be hidden within (part of the file, but not displayed).

To reduce the file size, use a compression call: `writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)`

* `remove_duplicates` enables/disables compression merging identical objects.
* `remove_unreferenced` enables/disables suppression of unused objects.

It is recommended to apply this process just before writing to the file/stream.

It depends on the PDF how well this works, but we have seen an 86% file
reduction (from 5.7 MB to 0.8 MB) within a real PDF.


## Removing Images

```{testcode}
from pypdf import PdfWriter

writer = PdfWriter(clone_from="example.pdf")

writer.remove_images()

writer.write("out-no-images.pdf")
```

## Reducing Image Quality

If we reduce the quality of the images within the PDF, we can **sometimes**
reduce the file size of the PDF overall. That depends on how well the reduced
quality image can be compressed.

```{testcode}
from pypdf import PdfWriter

writer = PdfWriter(clone_from="example.pdf")

for page in writer.pages:
    for img in page.images:
        img.replace(img.image, quality=80)

writer.write("out-low-quality.pdf")
```

## Lossless Compression

The easiest way to get a smaller file without changing how it looks is to call
{meth}`writer.compress <pypdf.PdfWriter.compress>` just before writing:

```{testcode}
from pypdf import PdfWriter

writer = PdfWriter(clone_from="example.pdf")
writer.compress()
writer.write("out-compressed.pdf")
```

It does the following:

* It merges identical objects and removes unused ones, as described in
  [Removing duplication](#removing-duplication).
* It compresses streams again where that makes them smaller, and it uses PNG predictors
  for images where that helps. You can trade size for speed with
  `writer.compress(level=6)`.
* With `pip install pypdf[compress]`, it also uses libdeflate, which takes about as long
  as zlib but gives smaller files, and it stores JPEG images in a smaller way that decodes
  to the same pixels. All PDF readers can decode both.
* It enables {attr}`writer.use_object_streams <pypdf.PdfWriter.use_object_streams>`,
  which stores most objects in compressed object streams.

The output needs a PDF 1.5 reader, and images are not re-encoded. For PDF/A files,
`compress()` follows the rules of the standard: Metadata stays uncompressed, and
PDF/A-1 files do not get object streams. It is not available for incremental updates.

To save more, you can give up image quality:

```{testcode}
writer = PdfWriter(clone_from="example.pdf")
writer.compress(image_quality=75, image_max_dpi=150)
writer.write("out-compressed-images.pdf")
```

`image_quality` stores images as JPEG, and `image_max_dpi` reduces images which have more
than that resolution even if they fill the page. Both need Pillow, and both change the
images, so please check the result: JPEG is fine for photos, but it can visibly change
charts and scanned text, and it alters images which carry measured data.

pypdf supports the FlateDecode filter which uses the zlib/deflate compression
method. It is a lossless compression, meaning the resulting PDF looks exactly
the same.

Deflate compression can be applied to a single page via
{meth}`page.compress_content_streams <pypdf._page.PageObject.compress_content_streams>`:

```{testcode}
from pypdf import PdfWriter

writer = PdfWriter(clone_from="example.pdf")

for page in writer.pages:
    page.compress_content_streams()  # This is CPU intensive!

writer.write("out-lossless.pdf")
```

`page.compress_content_streams` uses [`zlib.compress`](https://docs.python.org/3/library/zlib.html#zlib.compress)
and supports the `level` parameter: `level=0` means no compression,
`level=9` refers to the highest compression.

Using this method, we have seen a reduction by 70% (from 11.8 MB to 3.5 MB)
with a real PDF.

## Removing Sources

When a page is removed from the page list, its content will still be present in
the PDF file. This means that the data may still be used elsewhere.

Simply removing a page from the page list will reduce the page count but not the
file size. To exclude the content completely, the pages should not be
added to the PDF using the PdfWriter.append() function. Instead, only the
desired pages should be selected for inclusion
(note: [PR #1843](https://github.com/py-pdf/pypdf/pull/1843) will add a page
deletion feature).

There can be issues with poor PDF formatting, such as when all pages are linked
to the same resource. In such cases, dropping references to specific pages
becomes useless because there is only one source for all pages.

Cropping is an ineffective method for reducing the file size because it only
adjusts the viewboxes and not the external parts of the source image. Therefore,
the content that is no longer visible will still be present in the PDF.

## Going Further

The presentation [Putting a Squeeze on Your PDF](https://youtube.com/watch?v=tgOABUhVwFs) has other suggestions. One takeaway is that most of the significant size optimizations usually come from image and font modification. However, font optimization, such as replacing, merging, and subsetting, is not within the functionality of pypdf at the moment.
