# hipsnp

 <<< This is a development version, there is no any warranty that it works >>>

functions to handle SNP data, especially from the UKB.

```
>>> import hipsnp
>>> genotypes = hipsnp.vcf2genotype('snp_epilepsy.vcf')
>>> genotypes
                          sub-01  sub-02  sub-03  ...  sub-04  sub-05  sub-06
rs1231232,6:12364007_T_T      TT      TT      AC  ...      TC      AC      TT
rs1231231,6:12331231_C_A      CA      AC      CC  ...      CC      CC      CA

[2 rows x 487409 columns]
```

# resources


## SNP databases

https://www.ncbi.nlm.nih.gov/snp/

http://www.ensembl.org/Homo_sapiens

https://varsome.com


## Tools

https://www.well.ox.ac.uk/~gav/qctool/


## Info

https://eu.idtdna.com/pages/education/decoded/article/genotyping-terms-to-know

https://faculty.washington.edu/browning/intro-to-vcf.html

https://www.reneshbedre.com/blog/vcf-fields.html

https://www.garvan.org.au/research/kinghorn-centre-for-clinical-genomics/learn-about-genomics/for-gp/genetics-refresher-1/types-of-variants

