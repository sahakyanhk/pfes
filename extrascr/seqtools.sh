#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

#DIR=$(cd $(dirname $0); pwd)
#https://devhints.io/bash
#set -euo pipefail
#IFS=$'\n\t'

#calculate AA occurance 
#awk '$7>.80 {print $0}'  run1/progress.log |  tail -n +50000 |  awk '{print $10}' | fold -w1 | sort | uniq -c

muscle5="/data/saakyanh2/WD/PFES/pfes/bin/muscle5"
iqtree2="/data/saakyanh2/WD/PFES/pfes/bin/iqtree2"
FastTree="/data/saakyanh2/WD/PFES/pfes/bin/FastTree"
sr_filer="/data/saakyanh2/WD/PFES/pfes/bin/sr_filer.pl"
reformat="/data/saakyanh2/WD/PFES/pfes/bin/reformat.pl"
export PATH=/data/saakyanh2/WD/PFES/pfes/bin/mmseqs/bin/:$PATH

log=$1 

mkdir -p sequences_analyses
cd sequences_analyses

#grep -v "#" ../${log} | tail -n +2 | sort -u -k1,1 | sort -u -k12,12 | awk '$8 > 0.8 && $4 > 60 {print ">" $1, $12}' | tr ' ' '\n' > uniq_sequences.fasta
grep -v "#" ../${log} | tail -n +2 |  sort -u -k12,12 | awk '$8 > 0.8 && $4 > 80 {print ">" $1"_"$2, "\n" $12}' > uniq_sequences.fasta


echo "clustering sequences with --min-seq-id 90%"
mmseqs easy-cluster uniq_sequences.fasta c90 tmp --min-seq-id 0.9 -c 0.9 --cov-mode 0 > /dev/null
echo "clustering sequences with --min-seq-id 80%"
mmseqs easy-cluster uniq_sequences.fasta c80 tmp --min-seq-id 0.8 -c 0.8 --cov-mode 0 > /dev/null
echo "clustering sequences with --min-seq-id 70%"
mmseqs easy-cluster uniq_sequences.fasta c70 tmp --min-seq-id 0.7 -c 0.7 --cov-mode 0 > /dev/null
#echo "clustering sequences with --min-seq-id 60%"
#mmseqs easy-cluster uniq_sequences.fasta c60 tmp --min-seq-id 0.6 -c 0.6 --cov-mode 0 > /dev/null
echo "clustering sequences with --min-seq-id 50%"
mmseqs easy-cluster uniq_sequences.fasta c50 tmp --min-seq-id 0.5 -c 0.5 --cov-mode 0 > /dev/null

rm *_all_seqs.fasta *_cluster.tsv


#  ################ALL################
#  $muscle5 -align uniq_sequences.fasta -output uniq_sequences_aln.afa
#  $reformat fas a3m uniq_sequences_aln.afa uniq_sequences_aln.a3m
#  cat uniq_sequences_aln.afa |\
#   awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
#    sed -e '/^>/{N; s/\n/\t/;}' |\
#     tail -n +2 > uniq_sequences_aln.sr
#  
#  $sr_filer uniq_sequences_aln.sr -grcut=0.5 -hocut=0.1   | tr '\t' '\n' > uniq_sequences_aln_filtred.fasta

#$FastTree < uniq_sequences_aln_filtred.fasta > uniq_sequences.nwk
#$iqtree2 -s uniq_sequences_aln_filtred.fasta --prefix IQ100  -T 8

################clust80################
$muscle5 -align c90_rep_seq.fasta -output c90_aln.afa
$reformat fas a3m c90_aln.afa c90_aln.a3m

cat c90_aln.afa |\
 awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
  sed -e '/^>/{N; s/\n/\t/;}' |\
   tail -n +2 > c90_aln.sr

$sr_filer c90_aln.sr -grcut= 0.5 -hocut= 0.1   | tr '\t' '\n' > c90_aln_filtred.fasta

#$FastTree < c90_aln_filtred.fasta > c90.nwk
#$iqtree2 -s c90_aln_filtred.fasta --prefix IQ_c90  -T 8

################clust80################
$muscle5 -align c80_rep_seq.fasta -output c80_aln.afa
$reformat fas a3m c80_aln.afa c80_aln.a3m

cat c80_aln.afa |\
 awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
  sed -e '/^>/{N; s/\n/\t/;}' |\
   tail -n +2 > c80_aln.sr

$sr_filer c80_aln.sr -grcut= 0.5 -hocut= 0.1   | tr '\t' '\n' > c80_aln_filtred.fasta

#$FastTree < c80_aln_filtred.fasta > c80.nwk
#$iqtree2 -s c80_aln_filtred.fasta --prefix IQ_80  -T 8


#
#################clust70################
$muscle5 -align c70_rep_seq.fasta -output c70_aln.afa
$reformat fas a3m c70_aln.afa c70_aln.a3m

cat c70_aln.afa |\
 awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
  sed -e '/^>/{N; s/\n/\t/;}' |\
   tail -n +2 > c70_aln.sr

$sr_filer c70_aln.sr -grcut= 0.5 -hocut= 0.1   | tr '\t' '\n' > c70_aln_filtred.fasta

#$FastTree < c70_aln_filtred.fasta > c70.nwk
#$iqtree2 -s c70_aln_filtred.fasta -redo
#
#
#
################clust60################
#$muscle5 -align c60_rep_seq.fasta -output c60_aln.afa
#cat c60_aln.afa |\
# awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
#  sed -e '/^>/{N; s/\n/\t/;}' |\
#   tail -n +2 > c60_aln.sr
#
#../../pfes/sr_filer.pl c60_aln.sr -grcut= 0.5 -hocut= 0.1   | tr '\t' '\n' > c60_aln_filtred.fasta
#
#$FastTree < c60_aln_filtred.fasta > c60.nwk
#$iqtree2 -s c60_aln_filtred.fasta -redo
#
#
################clust60################
$muscle5 -align c50_rep_seq.fasta -output c50_aln.afa
$reformat fas a3m c50_aln.afa c50_aln.a3m

cat c50_aln.afa |\
 awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' |\
  sed -e '/^>/{N; s/\n/\t/;}' |\
   tail -n +2 > c50_aln.sr

$sr_filer c50_aln.sr -grcut= 0.5 -hocut= 0.1   | tr '\t' '\n' > c50_aln_filtred.fasta

#$FastTree < c50_aln_filtred.fasta > c50.nwk
#$iqtree2 -s c50_aln_filtred.fasta -redo
#
#
#