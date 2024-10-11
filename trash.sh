
docker run \
  -v /home/amadou/Data/.cache/.github-action-cache/control-dataset:/workdir/pymimic3/tests/data/control-dataset \
  -v /home/amadou/Data/.cache/.github-action-cache/mimiciii-demo:/workdir/pymimic3/tests/data/mimiciii-demo \
  -v /home/amadou/Data/.cache/.github-action-cache/semitemp:/workdir/pymimic3/tests/data/semitemp \
  -v /home/amadou/Data/.cache/.github-action-cache/bash-results:/workdir/pymimic3/tests/data/bash-results \
  -v /home/amadou/Data/.cache/.github-action-cache/pytest-results:/workdir/pymimic3/tests/data/pytest-results \
  tensorpod/pymimic3:feature_workflow_pytests \
  bash -c "cd etc && make testing=true "

  bash -c "echo \$(ls /workdir/pymimic3/tests/data/mimiciii-demo -la)"
  "cd etc ; make ; cd .. ; source etc/config_env.sh; pytest --no-cleanup -v tests/testz_setup_all_fixtures"


  bash -c "cd etc && make && cd .. && pytest --no-cleanup -v tests/testz_setup_all_fixtures" #>> /workdir/pymimic3/tests/data/bash-results/anus.txt 2>&1"

docker run\
 -v /home/amadou/Data/.cache//.github-action-cache/control-dataset:/workdir/tests/data/control-dataset\
 -v /home/amadou/Data/.cache//.github-action-cache/mimiciii-demo:/workdir/tests/data/mimiciii-demo\
 -v /home/amadou/Data/.cache//.github-action-cache/semitemp:/workdir/tests/data/semitemp\
 -v /home/amadou/Data/.cache//.github-action-cache/bash-results:/workdir/tests/data/bash-results\
 -v /home/amadou/Data/.cache//.github-action-cache/pytest-results:/workdir/tests/data/pytest-results\
 tensorpod/pymimic3:feature_workflow_pytests \  
bash -ic "mkdir -p /workdir/tests/data/pytest-results/6249f7e6aaa4a5b8faa5ad5cdb247f5cec03a025"

docker run\
 tensorpod/pymimic3:feature_workflow_pytests \  
bash -c "mkdir -p /workdir/tests/data/pytest-results/6249f7e6aaa4a5b8faa5ad5cdb247f5cec03a025"