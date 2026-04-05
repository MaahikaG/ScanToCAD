#!/bin/bash

SCANS=/Users/maahikagupta/scan_to_cad/ubuntu/scans

rm -f $SCANS/annotated.txt

echo "Waiting for scan to complete..."
while [ ! -f "$SCANS/annotated.txt" ]; do
    sleep 1
done

echo "annotated.txt found — running Point2CAD..."
docker run -it --rm \
  --user root \
  --platform linux/amd64 \
  --entrypoint python \
  -v /Users/maahikagupta/scan_to_cad/ubuntu/models/point2cad:/work/point2cad \
  -v $SCANS:/scans \
  toshas/point2cad:v1 \
  -m point2cad.main \
  --path_in /scans/annotated.txt \
  --path_out /scans/point2cad_out

echo "Done! Output in $SCANS/point2cad_out"
