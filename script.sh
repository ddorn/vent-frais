#!/bin/bash

set -e

mkdir -p shapes pdfs decks

function generate() {
        scale=$1
        density=$2
        # defaults to few-questions.csv
        questions=${3:-few-questions.csv}
        square_size=${4:-3}
        show=$5

        echo "Generating shapes for scale $scale, density $density, and square size $square_size"

        shape_file="shapes/scale$scale-newdensity$density-size$square_size.json"
        deck_file="decks/vent-frais-s$scale-d$density.deck"
        pdf_file="pdfs/vent-frais-s$scale-d$density.pdf"
        cache_dir="pdfs/vent-frais-rounding0-s$scale-d$density/"

        # Generate shapes if they don't exist
        if [ ! -f "$shape_file" ]; then
                ./cards.py shapes data/wind.npy "$shape_file" --scale "$scale" --min-density "$density" --size "$square_size"
        fi
        ./cards.py new -o "$deck_file" "$shape_file"
        ./cards.py csv --has-header \
                --question-col 2 \
                --category-col 8 \
                --category-map "Uplifting;Vulnerable;Debate;Future" \
                "$questions" \
                add-to-deck "$deck_file"
        ./cards.py pdf -o "$pdf_file" "$deck_file" --cache-dir "$cache_dir" --rounding 0 --format "collage-front"

        if [ "$show" = "show" ]; then
                echo "Showing $pdf_file"
                zathura "$pdf_file" > /dev/null 2>&1 &
        fi
}

# generate scale density csv square_size show
generate 10 5 data/questions-23-05-16.csv 12 show
# For the release, we use the following:
# scale: 10
# density: 5
# csv: data/questions-23-05-16.csv
# square_size: 12
# rounding: 0


# for scale in {16,32,64};do
#     for density in {64,128}; do
#         generate $scale $density;
#     done;
# done;


# Find all files int out/ generated before 4 am, and print their names
# find out/ -type f -mmin +240 -exec basename {} \; > out/old-files.txt

# Delete all files in pdfs/ with same name as files in out/old-files.txt
# while read -r line; do
#         rm pdfs/"$line"
# done < out/old-files.txt


# Show a question in random order from the svg in pdfs/vent-frais-test-s10-d5/
# open them all in the same svg reader (imv)
# find pdfs/vent-frais-rounding0-s10-d5/ -type f -name "*front.svg" | shuf | xargs imv -f -s full
