#!/bin/bash

for image in ./gothic_zoologist/data/gothic/**/*.png; do
  echo "Convert $image to jpg"
  convert "$image" "${image%.*}.jpg";
  rm -f "$image"
done