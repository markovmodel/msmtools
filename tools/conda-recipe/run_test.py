import os
import sys

import pytest

test_pkg = 'msmtools'
cover_pkg = test_pkg

junit_xml = os.path.join(os.getenv('CIRCLE_TEST_REPORTS', '.'), 'junit.xml')


pytest_args = ("-v --pyargs {test_pkg} "
               "--cov={cover_pkg} "
               "--cov-report=xml:{dest_report} "
               #"--doctest-modules "
               "--junit-xml={junit_xml} "
               .format(test_pkg=test_pkg, cover_pkg=cover_pkg,
                       junit_xml=junit_xml,
                       dest_report=os.path.join(os.path.expanduser('~/'), 'coverage.xml'),
                       )
               .split(' '))
print("args:", pytest_args)
res = pytest.main(pytest_args)

sys.exit(res)

