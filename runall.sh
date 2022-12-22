python -u ./occ-tests-binary-fdr.py | tee log.txt
python -u ./occ-tests-dataset-fdr.py | tee -a log.txt
python -u ./occ-tests-dist-fdr.py | tee -a log.txt
python -u ./occ-tests-binary-general.py | tee -a log.txt
python -u ./occ-tests-dataset-general.py | tee -a log.txt
python -u ./occ-tests-dist-general.py | tee -a log.txt
python -u ./occ-tests-binary-t1e.py | tee -a log.txt
python -u ./occ-tests-dataset-t1e.py | tee -a log.txt
python -u ./occ-tests-dist-t1e.py | tee -a log.txt
python -u ./occ-tests-simple-fdr.py | tee -a log.txt
tar -czf results.tar.gz results*/ | tee -a log.txt
echo "Done" | tee -a log.txt
