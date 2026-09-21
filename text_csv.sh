awk -F'\t' '
BEGIN { OFS="," }
NR==1 { print $8,$4,$3,$2,$9,$10,$7; next }
NR>1 && ($3 ~ /pm2.5_ug_m3/ || $3 ~  /temperature_C/ || $3 ~ /humidity_percent/) && ($2 ~ /SEN5X/) {
    gsub("'\''", "", $3)
	print $8,"=\"" $4 "\"",$3,$2,"=\"" $9 "\"",$10,$7
}
' /mnt/a/SimpleAQ/Jackson_Pike/combineD.tsv \
| awk NF \
 > /mnt/a/SimpleAQ/Jackson_Pike/OSU_D.csv

