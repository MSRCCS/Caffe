from __future__ import print_function
import os
import sys
import numpy as np
import argparse
import json
import re
import tarfile
import textwrap
from zipfile import ZipFile
from xml.etree import ElementTree

if sys.version_info >= (3, 0):
    from os import makedirs
else:
    import errno

    def makedirs(name, mode=511, exist_ok=False):
        try:
            os.makedirs(name, mode=mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            if not exist_ok:
                raise

_valid_extensions = [".jpg", ".png"]
_valid_phases = ["train", "test", "val"]
_synset_label_pattern = re.compile(r'^(?P<LABEL>n\d{8,9})(?P<EXT>_\d+)?(_(?P<META>\w+))?')
_coco_label_pattern = re.compile(r'^(?P<META>COCO_.*)_(?P<LABEL>\d{12,13})$')


def listarchive(path, is_root=True, extension_pattern='\.\w+', filter_func=None):
    """Similar to listdir but (in addition to directories) extract archives locally and recurse them
    :param path: the path to start the recursion
    :param is_root: if path is the first in recursion
    :param extension_pattern: regular expression pattern for file extensions
    :param filter_func: a filter to apply on directories and files, to limit the recursion search
    """
    if os.path.isdir(path):
        # Ignore hidden files and directories
        sub_paths = [s0 for s0 in os.listdir(path) if not s0.startswith(".") and not s0.startswith("$")]
        new_filter_func = filter_func
        if filter_func:
            filtered = [s0 for s0 in sub_paths if filter_func(s0)]
            if filtered:
                sub_paths = filtered
                new_filter_func = None  # once matched, recurse all the way
            elif len(sub_paths) > 1:
                # ignore non-matching paths with more than one item
                sub_paths = []
        for s0 in sub_paths:
            s0 = os.path.join(path, s0)
            for sub_path in listarchive(s0, is_root=False,
                                        extension_pattern=extension_pattern,
                                        filter_func=new_filter_func):
                yield sub_path
        return

    local_path, fname = os.path.split(path)
    basename, ext = os.path.splitext(fname)
    ext = ext.lower()
    shortname = basename
    while '.' in shortname:
        shortname, shortext = os.path.splitext(shortname)
        ext = shortext + ext

    if ext not in ['.tar', '.tar.gz', '.zip']:
        if re.match(extension_pattern, ext, flags=re.IGNORECASE):
            yield path
        return

    extracted_path = os.path.join(local_path, 'extracted_' + basename)
    if os.path.exists(extracted_path):
        for sub_path in listarchive(extracted_path, is_root=False,
                                    extension_pattern=extension_pattern,
                                    filter_func=filter_func):
            yield sub_path
        return

    # extract if not yet extracted
    makedirs(extracted_path, exist_ok=True)
    if ext in ['.tar', '.tar.gz']:
        with tarfile.open(path) as archive:
            archive.extractall(extracted_path)
    elif ext in ['.zip']:
        with ZipFile(path) as archive:
            archive.extractall(extracted_path)

    # recurse through the just-extracted path
    for sub_path in listarchive(extracted_path, is_root=False,
                                extension_pattern=extension_pattern,
                                filter_func=filter_func):
        yield sub_path

    # remove intermediate *extracted* archives
    if not is_root:
        os.remove(path)


def guess_phase(path):
    """Guess the phase from path
    :param path: file or directory path to guess the phase name from
    :rtype str
    """
    for elem in reversed(path.replace("\\", "/").split("/")):
        if not elem:
            continue
        elem_lower = elem.lower()
        for name in _valid_phases:
            if name in elem_lower:
                return elem
    return ""


def guess_label(path):
    """Guess the label from path
    :param path: file path to guess the label from
    :rtype (str, str, str)
    """

    for elem in reversed(path.replace("\\", "/").split("/")):
        if not elem:
            continue
        match = _synset_label_pattern.match(elem)
        if match:
            label = match.group('LABEL')
            full_label = label + match.group('EXT') or ''
            return label, full_label, match.group('META') or ''

    elem, _ = os.path.splitext(os.path.basename(path))
    match = _coco_label_pattern.match(elem)
    if match:
        full_label = label = match.group('LABEL')
        # TODO: Convert the label to synset equivalent
        return label, full_label, match.group('META')

    # Use immediate directory as label
    parent = os.path.dirname(path)
    label = os.path.basename(parent)
    return label, label, label


def gather_images(root_path, imagedata, counts, phase=None, max_keep_per_label=np.inf):
    """Create image structure from images in root_path
    Example:
    /root_path/training/n04422727/blue_cheese.jpg
    /root_path/training/n04422727_43.jpg
    /root_path/training/cheeses/n04422727_41.jpg
    /root_path/n04422727_41_training.jpg
    /root_path/training/n04422727_42_bluecheese.jpg
    /root_path/training/COCO_val2014_000000006818.jpg
    """
    for s0 in os.listdir(root_path):
        if s0.startswith(".") or s0.startswith("$"):
            # Ignore hidden files and directories
            continue
        s0 = os.path.join(root_path, s0)
        if not phase:
            phase = guess_phase(s0)
        if os.path.isdir(s0):
            gather_images(s0, imagedata, counts, phase, max_keep_per_label=max_keep_per_label)
            continue
        _, file_extension = os.path.splitext(s0)
        if file_extension.lower() not in _valid_extensions:
            continue

        label, full_label, meta = guess_label(s0)
        if label not in counts:
            counts[label] = 1
        elif counts[label] >= max_keep_per_label:
            continue
        else:
            counts[label] += 1

        if not phase:
            for phase in imagedata.keys():
                for name in _valid_phases:
                    if name in phase.lower():
                        break
            if not phase:
                phase = "training"
            print("Phase {} was assumed when processing {}".format(phase, s0))

        if phase not in imagedata:
            imagedata[phase] = [(s0, label, full_label, meta)]
        else:
            imagedata[phase].append((s0, label, full_label, meta))


def get_xml_rects(path, label):
    """Get annotation from VOC-style XML
    """
    rects = []
    tree = ElementTree.parse(path)
    for obj in tree.findall('object'):
        name = obj.find('name').text
        if name != label:
            print('Ignore label "{}" != {} in {}'.format(name, label, path))
            continue
        diff = int(obj.find('difficult').text)
        if diff:
            print('Ignore difficult label "{}" in {}'.format(label, path))
            continue
        for bndbox in obj.findall('bndbox'):
            rect = [int(bndbox.find(k).text) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
            rects.append(rect)

    return rects


def get_boxes(full_label, label, meta, annotations):
    """Get the list of boxes for a label
    :rtype list
    """
    if 'COCO' in meta:
        # TODO: implement using nltk wordnet
        return

    def voc_ann_filter(elem):
        if label in elem:
            return True

    for ann_path in annotations:
        for path in listarchive(ann_path, extension_pattern='\.xml', filter_func=voc_ann_filter):
            fname = os.path.basename(path)
            if fname.startswith(full_label):
                return [{'class': label, 'rect': rect} for rect in get_xml_rects(path, label)]

    boxes = [{'class': label, 'rect': [0, 0, 0, 0]}]  # full image as a box
    return boxes


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Process images in a path recursively and prepare TSV files.',
        epilog=textwrap.dedent('''Example:
convert_to_tsv.py d:/data/imagenet/ -a d:/data/imagenet/Annotation.tar.gz'''))

    parser.add_argument('-k', '--keep', help='Maximum number of images to keep for each label',
                        type=float, default=np.inf)
    parser.add_argument('-a', '--annotation', action='append', required=True,
                        help='Annotation archive file, or directory (can be specified multiple times)')
    parser.add_argument('root_path', metavar='PATH', help='path to the images dataset')

    if len(sys.argv) == 1:
        parser.print_help()
        raise Exception("Required input not provided")

    args = parser.parse_args()
    root_path = args.root_path
    max_keep_per_label = args.keep

    images = {}
    counts = {}
    gather_images(root_path, images, counts, max_keep_per_label=max_keep_per_label)

    for phase, vs in images.items():
        with open(os.path.join(root_path, phase + '.lineidx'), "w") as idx_file:
            with open(os.path.join(root_path, phase + '.tsv'), "w") as tsv_file:
                for v in vs:
                    idx_file.write("{}\n".format(tsv_file.tell()))
                    path, label, full_label, meta = v
                    relpath = os.path.relpath(path, root_path).replace("\\", "/")
                    boxes = get_boxes(full_label, label, meta, args.annotation)
                    tsv_file.write("{}\t{}\t{}\n".format(full_label, json.dumps(boxes), relpath))

    return images, counts

if __name__ == '__main__':
    main_results = main()
