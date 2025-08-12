import pytest
import cpmpy as cp
from pycona.find_constraint import FindC, FindC2
from pycona.ca_environment.active_ca import ActiveCAEnv
from pycona.find_constraint.findc_obj import findc_obj_splithalf, findc_obj_proba
from pycona.benchmarks.golomb import construct_golomb
import pycona as ca

algorithms = [FindC(), FindC2()]
fast_algorithms = [FindC()]  # Use only FindC for fast tests

class TestFinC:
    @pytest.mark.fast
    def test_findc_query_generation(self):
        """Test query generation in FindC"""
        ca_env = ActiveCAEnv()
        findc = FindC(ca_env=ca_env)
        
        # Create test variables
        x = cp.intvar(1, 10, name="x")
        y = cp.intvar(1, 10, name="y")
        
        # Create constraints
        L = [x <= y]  # learned constraints
        delta = [x < y, x >= y, x == y]  # candidate constraints
        
        # Test query generation
        assert findc.generate_findc_query(L, delta)

    @pytest.mark.fast
    def test_findc2_query_generation(self):
        """Test query generation in FindC2"""
        ca_env = ActiveCAEnv()
        findc2 = FindC2(ca_env=ca_env)
        
        # Create test variables
        x = cp.intvar(1, 10, name="x")
        y = cp.intvar(1, 10, name="y")
        
        # Create constraints
        L = [x <= y]  # learned constraints
        delta = [x < y, x >= y, x == y]  # candidate constraints
        
        # Test query generation
        assert findc2.generate_findc_query(L, delta)

    @pytest.mark.fast
    def test_findc_objective_functions(self):
        """Test objective function changes in FindC"""
        findc = FindC()
        
        # Test probability-based objective
        findc.obj = findc_obj_proba
        assert findc.obj == findc_obj_proba
        
        # Test split-half objective
        findc.obj = findc_obj_splithalf
        assert findc.obj == findc_obj_splithalf

    def test_findc2_with_golomb4(self):
        """Test FindC with a Golomb ruler of order 4"""
        ca_env = ActiveCAEnv(findc=FindC2())
        alg = ca.QuAcq(ca_env)
        
        # Create Golomb ruler instance of order 4
        instance, oracle = construct_golomb(n_marks=4)
        
        li = alg.learn(instance, oracle)
        
        # oracle model imply learned?
        oracle_not_learned = cp.Model(oracle.constraints)
        oracle_not_learned += cp.any([~c for c in li._cl])
        assert not oracle_not_learned.solve()

        # learned model imply oracle?
        learned_not_oracle = cp.Model(li._cl)
        learned_not_oracle += cp.any([~c for c in oracle.constraints])
        assert not learned_not_oracle.solve()

        # test growacq
        alg = ca.GrowAcq(ca_env, alg)
        li2 = alg.learn(instance, oracle)

        # oracle model imply learned?
        oracle_not_learned = cp.Model(oracle.constraints)
        oracle_not_learned += cp.any([~c for c in li2._cl])
        assert not oracle_not_learned.solve()

        # learned model imply oracle?
        learned_not_oracle = cp.Model(li2._cl)
        learned_not_oracle += cp.any([~c for c in oracle.constraints])
        assert not learned_not_oracle.solve()


    
