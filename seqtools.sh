#!/bin/bash
set -e

muscle5="/data/saakyanh2/WD/PFES/pfes/bin/muscle5"
iqtree2="/data/saakyanh2/WD/PFES/pfes/bin/iqtree2"

log=$1 

mkdir -p sequences
cd sequences

tail -n +3 ../${log} | sort -u -k1,1 | cut  -f 11 | sort | uniq -c | awk '{print ">" NR, $2}' | tr ' ' '\n' > uniqseqs.fasta

echo "clustering sequences with min identity of 90%"
mmseqs easy-cluster uniqseqs.fasta c90 tmp --min-seq-id 0.9 -c 0.9 --cov-mode 0 > /dev/null
echo "clustering sequences with min identity of 80%"
mmseqs easy-cluster uniqseqs.fasta c80 tmp --min-seq-id 0.8 -c 0.8 --cov-mode 0 > /dev/null
echo "clustering sequences with min identity of 70%"
mmseqs easy-cluster uniqseqs.fasta c70 tmp --min-seq-id 0.7 -c 0.7 --cov-mode 0 > /dev/null
echo "clustering sequences with min identity of 60%"
mmseqs easy-cluster uniqseqs.fasta c70 tmp --min-seq-id 0.6 -c 0.6 --cov-mode 0 > /dev/null
echo "clustering sequences with min identity of 50%"
mmseqs easy-cluster uniqseqs.fasta c50 tmp --min-seq-id 0.5 -c 0.5 --cov-mode 0 > /dev/null

$muscle5 -align c50_rep_seq.fasta -output c50_aln.afa
$iqtree2 -s c50_aln.afa -t tree


#tr ' ' '\t'


