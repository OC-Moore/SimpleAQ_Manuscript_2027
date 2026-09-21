#run with dos2unix then bash and not sh

output_file1="combineA.tsv"
output_file2="combineB.tsv"
output_file3="combineD.tsv"

# Find the first .gz file and extract its header
header_file=$(find /mnt/a/SimpleAQ/Jackson_Pike/cmhz6ewe60003s60ds0e5vovf -type f -name '*.tsv.gz' | head -n 1)
gunzip -c "$header_file" | head -n 1 > "$output_file1"

# Find the remaining .gz files and concatenate their contents (excluding headers)
find /mnt/a/SimpleAQ/Jackson_Pike/cmhz6ewe60003s60ds0e5vovf -type f -name '*.tsv.gz' | while read -r gz_file; do
  gunzip -c "$gz_file" | tail -n +2 >> "$output_file1"
done
header_file=$(find /mnt/a/SimpleAQ/Jackson_Pike/cmhz6q40n0005s60dg3zv0f9s -type f -name '*.tsv.gz' | head -n 1)
gunzip -c "$header_file" | head -n 1 > "$output_file2"

find /mnt/a/SimpleAQ/Jackson_Pike/cmhz6q40n0005s60dg3zv0f9s -type f -name '*.tsv.gz' | while read -r gz_file; do
  gunzip -c "$gz_file" | tail -n +2 >> "$output_file2"
done
header_file=$(find /mnt/a/SimpleAQ/Jackson_Pike/cmi7tqce10004s60ddvtjlkdx -type f -name '*.tsv.gz' | head -n 1)
gunzip -c "$header_file" | head -n 1 > "$output_file3"

find /mnt/a/SimpleAQ/Jackson_Pike/cmi7tqce10004s60ddvtjlkdx -type f -name '*.tsv.gz' | while read -r gz_file; do
  gunzip -c "$gz_file" | tail -n +2 >> "$output_file3"
done