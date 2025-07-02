#!/bin/bash

du -sh /mnt/StorageMedia/dzambala_data/
rdfind -makehardlinks true /mnt/StorageMedia/dzambala_data/
rm results.txt
du -sh /mnt/StorageMedia/dzambala_data/
