#!/bin/bash

du -sh /mnt/StorageMedia/dzambala_data/
rdfind -makehardlinks true -removeidentinode false /mnt/StorageMedia/dzambala_data/
rm results.txt
du -sh /mnt/StorageMedia/dzambala_data/
