#!/bin/bash
prev=0
after=0
for file in *.mp3
do
	prev=$((prev+1))
done

for file in *.mp3
do
	after=$((after+1))
	filename=$(echo "$file" | cut -f 1 -d '.')
	mpg123 -w ./${filename}.wav ./${filename}.mp3
	rm -f ./${filename}.mp3
done

echo $prev $after