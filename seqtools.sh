#!/bin/bash
set -e

muscle5="/data/saakyanh2/WD/PFES/pfes/bin/muscle5"
iqtree2="/data/saakyanh2/WD/PFES/pfes/bin/iqtree2"
FastTree="/data/saakyanh2/WD/PFES/pfes/bin/FastTree"

log=$1 

mkdir -p sequences
cd sequences

tail -n +3 ../${log} | sort -u -k1,1 | sort -u -k11,11 | awk '{print ">" $1, $11}' | tr ' ' '\n' > uniq_sequences.fasta

echo "clustering sequences with min identity of 90%"
mmseqs easy-cluster uniqseqs.fasta c90 tmp --min-seq-id 0.9 -c 0.9 --cov-mode 0 > /dev/null
echo "clustering sequences with min identity of 80%"
mmseqs easy-cluster uniqseqs.fasta c80 tmp --min-seq-id 0.8 -c 0.8 --cov-mode 0 > /dev/null
echo "clustering sequences with min identity of 70%"
mmseqs easy-cluster uniqseqs.fasta c70 tmp --min-seq-id 0.7 -c 0.7 --cov-mode 0 > /dev/null
echo "clustering sequences with min identity of 60%"
mmseqs easy-cluster uniqseqs.fasta c60 tmp --min-seq-id 0.6 -c 0.6 --cov-mode 0 > /dev/null
echo "clustering sequences with min identity of 50%"
mmseqs easy-cluster uniqseqs.fasta c50 tmp --min-seq-id 0.5 -c 0.5 --cov-mode 0 > /dev/null

rm *_all_seqs.fasta *_cluster.tsv

################clust80################
$muscle5 -align c80_rep_seq.fasta -output c80_aln.afa
cat c80_aln.afa |\
 awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
  sed -e '/^>/{N; s/\n/\t/;}' |\
   tail -n +2 > c80_aln.sr

../../pfes/sr_filer.pl c80_aln.sr -grcut= 0.5 -hocut= 0.1   | tr '\t' '\n' > c80_aln_filtred.fasta

$FastTree < c80_aln_filtred.fasta > c80.nwk
$iqtree2 -s c80_aln_filtred.fasta 



################clust70################
$muscle5 -align c70_rep_seq.fasta -output c70_aln.afa
cat c70_aln.afa |\
 awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
  sed -e '/^>/{N; s/\n/\t/;}' |\
   tail -n +2 > c70_aln.sr

../../pfes/sr_filer.pl c70_aln.sr -grcut= 0.5 -hocut= 0.1   | tr '\t' '\n' > c70_aln_filtred.fasta

$FastTree < c70_aln_filtred.fasta > c70.nwk
$iqtree2 -s c70_aln_filtred.fasta -redo



###############clust60################
$muscle5 -align c60_rep_seq.fasta -output c60_aln.afa
cat c60_aln.afa |\
 awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
  sed -e '/^>/{N; s/\n/\t/;}' |\
   tail -n +2 > c60_aln.sr

../../pfes/sr_filer.pl c60_aln.sr -grcut= 0.5 -hocut= 0.1   | tr '\t' '\n' > c60_aln_filtred.fasta

$FastTree < c60_aln_filtred.fasta > c60.nwk
$iqtree2 -s c60_aln_filtred.fasta -redo


###############clust60################
$muscle5 -align c50_rep_seq.fasta -output c50_aln.afa
cat c50_aln.afa |\
 awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
  sed -e '/^>/{N; s/\n/\t/;}' |\
   tail -n +2 > c50_aln.sr

../../pfes/sr_filer.pl c50_aln.sr -grcut= 0.5 -hocut= 0.1   | tr '\t' '\n' > c50_aln_filtred.fasta

$FastTree < c50_aln_filtred.fasta > c50.nwk
$iqtree2 -s c50_aln_filtred.fasta -redo

