#!/bin/bash

for image in /app/gothic_zoologist/data/prepare/**/*.png; do
  echo "Convert $image to jpg"

  convert "$image" \
      -resize 900x900^ \
      -gravity center \
      -extent 900x900 \
       "${image%.*}.jpg";

  convert "$image" \
      -resize 3840x2160^ \
      -gravity center \
      -extent 900x900 \
       "${image%.*}_2.jpg";

  convert "$image" \
      -resize 2880x1620^ \
      -gravity center \
      -extent 900x900 \
       "${image%.*}_1.jpg";

  rm -f "$image"
done