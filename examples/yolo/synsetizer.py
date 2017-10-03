import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown

_brown_ic = wn.ic(brown)


def nn_suggestion(phrase):
    """Suggest the name in a phrase
    """
    if " " not in phrase:
        return

    names = [n for (n, t) in nltk.pos_tag(phrase.split()) if t == 'NN']
    if len(names) > 1:
        print('Warning: "{}" has multiple NN to choose from'.format(phrase))
    for nn in names:
        yield nn


def name_suggestions(phrase):
    """Go through the name suggestions
    """
    yield phrase
    if ' ' in phrase:
        yield '_'.join(phrase.split())  # Wordnet substitutes space with underscore
    yield phrase.replace(" ", "")
    for nn in nn_suggestion(phrase):
        yield nn


def _synset_offset(name, parent_synset=None, parent=None):
    """synset ofset of name that descends from (or is closest to) the parent synset
    Original paper keeps shortest paths to root, but that is not useful: e.g. food, rhea, bench ...
    """

    for label in name_suggestions(name):
        synsets = wn.synsets(label, pos='n')
        if not synsets:
            continue

        paths = []
        if parent_synset:
            # find how similar this is to the parent
            paths_len = [-(ss.lin_similarity(parent_synset, _brown_ic) or 0) for ss in synsets]
            paths = [[ss] for ss in synsets]
        elif parent:
            # find if parent is mentioned in the definition
            paths = [ss.hypernym_paths() for ss in synsets if parent in ss.definition()]
            paths = [path for subpaths in paths for path in subpaths]
            paths_len = np.array([len(p) for p in paths])

        if paths:
            shortest_idx, = np.where(paths_len - np.min(paths_len) < 0.001)
            synsets = [paths[idx][-1] for idx in shortest_idx]

        if len(synsets) == 1:
            return synsets[0]

        # count lemmas and use the first one with highest frequency
        freqs = [np.sum([l.count() for l in ss.lemmas()]) for ss in synsets]
        synset = synsets[np.argmax(freqs)]
        print("Warning: {} chosen for n: {} p: {} among: {}".format(synset, name, parent_synset, synsets))
        return synset


class Synsetizer(object):
    def __init__(self):
        self._labels = {}

    def ss2of(self, key, synset):
        """Convert synset to offset
        """
        off = 'n' + wn.ss2of(synset)[:8]
        self._labels[key] = off
        return off

    def synset_offset(self, name, parent):
        """Find the label for name and its parent
        :param name: category name
        :param parent: super-category name
        :rtype: str
        """
        key = (name, parent)
        if key in self._labels:
            return self._labels[key]

        # first try those that descend from the parent in the wordnet
        synset = None
        for parent_name in name_suggestions(parent):
            for parent_synset in wn.synsets(parent_name, pos='n'):
                synset = _synset_offset(name, parent_synset)
                if synset:
                    break
            if synset:
                break

        # then try without parent (or with just simple relation)
        if not synset:
            synset = _synset_offset(name, parent=parent)
            if synset:
                print("Warning: {} of n: {} did not descend from p: {}".format(synset, name, parent))

        if not synset:
            synset = _synset_offset(parent)
            if synset:
                print("Warning: n: {} substituted with {} of p: {}".format(name, synset, parent))

        if not synset:
            raise Exception('Could not find any offset for n: {} p: {}'.format(name, parent))

        return self.ss2of(key, synset)
