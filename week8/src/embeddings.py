"""Leave-one-pair-out directions in a unit-normalized embedding table."""
import numpy as np

PAIRS={
 'Country to capital':'iran tehran turkey ankara russia moscow china beijing france paris sweden stockholm japan tokyo peru lima kenya nairobi',
 'Male to female':'king queen man woman he she brother sister father mother son daughter uncle aunt boy girl husband wife',
 'Adjective to comparative':'big bigger small smaller fast faster dark darker cold colder old older strong stronger long longer tall taller',
}
PAIRS={name:list(zip(words.split()[::2],words.split()[1::2])) for name,words in PAIRS.items()}


def offset_cosines(offsets):
    others=(offsets.sum(axis=0)-offsets)/(len(offsets)-1)
    return np.sum(offsets*others,axis=1)/(np.linalg.norm(offsets,axis=1)*np.linalg.norm(others,axis=1))


def relation_cosines(table,ids):
    rows=table[ids]
    rows=rows/np.linalg.norm(rows,axis=-1,keepdims=True)
    return offset_cosines(rows[:,1]-rows[:,0])
