import pyomo.environ as pyo
from pyomo.contrib import appsi

def create_model():
    """
    Builds an abstract model on top of the ecoinvent database.

    Returns:
        AbstractModel: The Pyomo abstract model for optimization.
    """
    # model = pyomo_base_model()
    # model.build_base_model()
    # ATTN: Since the code is written to be tested with the supply model formulation this works, if only the base_model is run then the test fail.
    # ATTN: A manual test or a change in the test structure is needed to see of th the base_model also works.
    model = pyomo_supply_model()
    model.build_supply_model()
    return model

class pyomo_base_model(pyo.AbstractModel):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def _declare_sets(self):
        # Sets
        self.PRODUCT = pyo.Set(doc='Set of intermediate products (or technosphere exchanges), indexed by i')
        self.PROCESS = pyo.Set(doc='Set of processes (or activities), indexed by j')
        self.ENV_COST = pyo.Set(doc='Set of environmental cost flows, indexed by e')
        self.INDICATOR = pyo.Set(doc='Set of impact assessment indicators, indexed by h')
        self.INV = pyo.Set(doc='Set of intervention flows, indexed by g')
        self.ENV_COST_PROCESS = pyo.Set(within=self.ENV_COST * self.PROCESS * self.INDICATOR, doc='Relation set between environmental cost flows and processes')
        self.ENV_COST_IN = pyo.Set(self.INDICATOR, within=self.ENV_COST)
        self.ENV_COST_OUT = pyo.Set(self.ENV_COST * self.INDICATOR, within=self.PROCESS)
        self.PROCESS_IN = pyo.Set(self.PROCESS, within=self.PRODUCT)
        self.PROCESS_OUT = pyo.Set(self.PRODUCT, within=self.PROCESS)
        self.PRODUCT_PROCESS = pyo.Set(within=self.PRODUCT * self.PROCESS, doc='Relation set between intermediate products and processes')
        self.INV_PROCESS = pyo.Set(within=self.INV * self.PROCESS, doc='Relation set between environmental flows and processes')
        self.INV_OUT = pyo.Set(self.INV, within=self.PROCESS)
        # Building rules for sets
        # ATTN: I don't like the BuildAction approach coded like this, because 
        #       1. why are we introducing "Env_in_out" and other variables? 
        #       2. why is the constrcutor method seperated, creates ambigious code snippets.
        #       3. It is not clear what changes in the model, I guess its just that the sets are populated?
        #       4. Why does this not happen in the set definitions, e.g., using the filter arguemnt?
        self.Env_in_out = pyo.BuildAction(rule=self._populate_env)
        self.Process_in_out = pyo.BuildAction(rule=self._populate_in_and_out)
        self.Inv_in_out = pyo.BuildAction(rule=self._populate_inv)

    # Rule functions
    # ATTN: I have not worked with pyo.BuildAction, hence I dont know if these methods can be static methods.
    @staticmethod
    def _populate_env(model):
        """Relates the environmental flows to the processes."""
        for i, j, h in model.ENV_COST_PROCESS:
            if i not in model.ENV_COST_IN[h]:
                model.ENV_COST_IN[h].add(i)
            model.ENV_COST_OUT[i, h].add(j)
    @staticmethod
    def _populate_in_and_out(model):
        """Relates the inputs of an activity to its outputs."""
        for i, j in model.PRODUCT_PROCESS:
            model.PROCESS_OUT[i].add(j)
            model.PROCESS_IN[j].add(i)
    @staticmethod
    def _populate_inv(model):
        """Relates the impacts to the environmental flows"""
        for a, j in model.INV_PROCESS:
            model.INV_OUT[a].add(j)

    def _declare_parameters(self):
        # Parameters
        self.UPPER_LIMIT = pyo.Param(self.PROCESS, mutable=True, within=pyo.Reals, doc='Maximum production capacity of process j')
        self.LOWER_LIMIT = pyo.Param(self.PROCESS, mutable=True, within=pyo.Reals, doc='Minimum production capacity of process j')
        self.UPPER_INV_LIMIT = pyo.Param(self.INV, mutable=True, within=pyo.Reals, doc='Maximum intervention flow g')
        self.UPPER_IMP_LIMIT = pyo.Param(self.INDICATOR, mutable=True, within=pyo.Reals, doc='Maximum impact on category h')
        self.ENV_COST_MATRIX = pyo.Param(self.ENV_COST_PROCESS, mutable=True, doc='Enviornmental cost matrix Q*B describing the environmental cost flows e associated to process j')
        self.INV_MATRIX = pyo.Param(self.INV_PROCESS, mutable=True, doc='Intervention matrix B describing the intervention flow g entering/leaving process j')
        self.FINAL_DEMAND = pyo.Param(self.PRODUCT, mutable=True, within=pyo.Reals, doc='Final demand of intermediate product flows (i.e., functional unit)')
        self.TECH_MATRIX = pyo.Param(self.PRODUCT_PROCESS, mutable=True, doc='Technology matrix A describing the intermediate product i produced/absorbed by process j')
        self.WEIGHTS = pyo.Param(self.INDICATOR, mutable=True, within=pyo.NonNegativeReals, doc='Weighting factors for the impact assessment indicators in the objective function')

    def _declare_variables(self):
        # Variables
        self.impacts = pyo.Var(self.INDICATOR, bounds=(-1e24, 1e24), doc='Environmental impact on indicator h evaluated with the established LCIA method')
        self.scaling_vector = pyo.Var(self.PROCESS, bounds=(-1e24, 1e24), doc='Activity level of each process to meet the final demand')
        self.inv_vector = pyo.Var(self.INV, bounds=(-1e24, 1e24), doc='Intervention flows')

    def _declare_constraints(self):
        # Constraints
        self.FINAL_DEMAND_CNSTR = pyo.Constraint(self.PRODUCT, rule=self._demand_constraint)
        self.IMPACTS_CNSTR = pyo.Constraint(self.INDICATOR, rule=self._impact_constraint)
        self.INVENTORY_CNSTR = pyo.Constraint(self.INV, rule=self._inventory_constraint)
        self.UPPER_CNSTR = pyo.Constraint(self.PROCESS, rule=self._upper_constraint)
        self.LOWER_CNSTR = pyo.Constraint(self.PROCESS, rule=self._lower_constraint)
        self.INV_CNSTR = pyo.Constraint(self.INV, rule=self._upper_env_constraint)
        self.IMP_CNSTR = pyo.Constraint(self.INDICATOR, rule=self._upper_imp_constraint)

    def _declare_objectives(self):
        # Objective function
        self.OBJ = pyo.Objective(sense=pyo.minimize, rule=self._objective_function)

    def build_base_model(self):
        self._declare_sets()
        self._declare_parameters()
        self._declare_variables()
        self._declare_constraints()
        self._declare_objectives()

    @staticmethod
    def _demand_constraint(model, i):
        """Fixes a value in the demand vector"""
        # ATTN: Deleted the slack from here now.
        return sum(model.TECH_MATRIX[i, j] * model.scaling_vector[j] for j in model.PROCESS_OUT[i]) == model.FINAL_DEMAND[i]
    @staticmethod
    def _impact_constraint(model, h):
        """Calculates all the impact categories"""
        return model.impacts[h] == sum(sum(model.ENV_COST_MATRIX[i, j, h] * model.scaling_vector[j] for j in model.ENV_COST_OUT[i, h]) for i in model.ENV_COST_IN[h])
    @staticmethod
    def _inventory_constraint(model, g):
        """Calculates the environmental flows"""
        return model.inv_vector[g] == sum(model.INV_MATRIX[g, j] * model.scaling_vector[j] for j in model.INV_OUT[g])
    @staticmethod
    def _upper_constraint(model, j):
        """Ensures that variables are within capacities (Maximum production constraint) """
        return model.scaling_vector[j] <= model.UPPER_LIMIT[j]
    @staticmethod
    def _lower_constraint(model, j):
        """ Minimum production constraint """
        return model.scaling_vector[j] >= model.LOWER_LIMIT[j]
    @staticmethod
    def _upper_env_constraint(model, g):
        """Ensures that variables are within capacities (Maximum production constraint) """
        return model.inv_vector[g] <= model.UPPER_INV_LIMIT[g]
    @staticmethod
    def _upper_imp_constraint(model, h):
        """ Imposes upper limits on selected impact categories """
        return model.impacts[h] <= model.UPPER_IMP_LIMIT[h]
    # ATTN: I would probably seperate the single objective formulation from the multi objective formulation.
    @staticmethod
    def _objective_function(model):
        """Objective is a sum over all indicators with weights. Typically, the indicator of study has weight 1, the rest 0"""
        return sum(model.impacts[h] * model.WEIGHTS[h] for h in model.INDICATOR)
    
class pyomo_supply_model(pyomo_base_model):
    """  
    Description:
        the special case where the
        practitioner wants to specify the total production quantities (supply)
        rather than the amount available for use outside the system (demand).
        This case can be modelled by specifying identical lower and upper bounds
        ($s_j^{low} = s_j^{high}$) on a scaling vector entry equal to the desired
        amount produced. For this particular process $j$ the corresponding product
        slack variable $slack_i^{high}$ must be set to a large value. Otherwise,
        it takes the value 0 and regular demand specification is performed for
        the product $i$ being the reference product of process $j$.
    """
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
    
    def build_supply_model(self):
        self.build_base_model()
        self._change_to_supply_model()

    def _change_to_supply_model(self):
        # Add SUPPLY as parameter for the slack variable bound
        # ATTN: do we still need the SUPPLY formulation if we seperate the supply and demand version of the problem into two different optimization problems?
        self.SUPPLY = pyo.Param(self.PRODUCT, mutable=True, within=pyo.Binary, doc='Binary parameter which specifies whether or not a supply has been specified instead of a demand')
        # Add slack as a variable
        self.slack = pyo.Var(self.PRODUCT, bounds=(0, 1e24), doc='Supply slack variables')
        # Add the variable bound of the slack as a constraint
        self.SLACK_CNSTR = pyo.Constraint(self.PRODUCT, rule=self._slack_constraint)
        # Change the final demand constraint to include the slack formulation
        self.FINAL_DEMAND_CNSTR = pyo.Constraint(self.PRODUCT, rule=self._demand_constraint_with_slack)

    @staticmethod
    # ATTN: do we still need this formulation if we seperate the supply and demand version of the problem into two different optimization problems. We could just define the variable bound when defining the slack variable
    def _slack_constraint(model, j):
        """ Slack variable upper limit for activities where supply is specified instead of demand """
        return model.slack[j] <= 1e20 * model.SUPPLY[j]
    @staticmethod
    def _demand_constraint_with_slack(model, i):
        """Fixes a value in the demand vector"""
        return sum(model.TECH_MATRIX[i, j] * model.scaling_vector[j] for j in model.PROCESS_OUT[i]) == model.FINAL_DEMAND[i] + model.slack[i]
    
def calculate_methods(instance, lci_data, methods):
    """
    Calculates the impacts if a method with weight 0 has been specified.

    Args:
        instance: The Pyomo model instance.
        lci_data (dict): LCI data containing matrices and mappings.
        methods (dict): Methods for environmental impact assessment.

    Returns:
        instance: The updated Pyomo model instance with calculated impacts.
    """
    import scipy.sparse as sparse
    import numpy as np
    matrices = lci_data['matrices']
    intervention_matrix = lci_data['intervention_matrix']
    matrices = {h: matrices[h] for h in matrices if str(h) in methods}
    env_cost = {h: sparse.csr_matrix.dot(matrices[h], intervention_matrix) for h in matrices}
    try:
        scaling_vector = np.array([instance.scaling_vector[x].value for x in instance.scaling_vector])
    except:
        scaling_vector = np.array([instance.scaling_vector[x] for x in instance.scaling_vector])

    impacts = {}
    for h in matrices:
        impacts[h] = sum(env_cost[h].dot(scaling_vector))

    # Check if instance.impacts_calculated exists
    if hasattr(instance, 'impacts_calculated'):
        # Update values if it already exists
        for h in impacts:
            instance.impacts_calculated[h].value = impacts[h]
    else:
        # Create instance.impacts_calculated
        instance.impacts_calculated = pyo.Var([h for h in impacts], initialize=impacts)
    return instance

def calculate_inv_flows(instance, lci_data):
    """
    Calculates elementary flows post-optimization.

    Args:
        instance: The Pyomo model instance.
        lci_data (dict): LCI data containing matrices and mappings.

    Returns:
        instance: The updated Pyomo model instance with calculated intervention flows.
    """
    import numpy as np
    intervention_matrix = lci_data['intervention_matrix']
    try:
        scaling_vector = np.array([instance.scaling_vector[x].value for x in instance.scaling_vector])
    except:
        scaling_vector = np.array([instance.scaling_vector[x] for x in instance.scaling_vector])
    flows = intervention_matrix.dot(scaling_vector)
    instance.inv_flows = pyo.Var(range(0, intervention_matrix.shape[0]), initialize=flows)
    return instance



def instantiate(model_data):
    """
    Builds an instance of the optimization model with specific data and objective function.

    Args:
        model_data (dict): Data dictionary for the optimization model.

    Returns:
        ConcreteModel: The instantiated Pyomo model.
    """
    print('Creating Instance')
    model = create_model()
    problem = model.create_instance(model_data, report_timing=False)
    print('Instance created')
    return problem


def solve_model(model_instance, gams_path=None, solver_name=None, options=None):
    """
    Solves the instance of the optimization model.

    Args:
        model_instance (ConcreteModel): The Pyomo model instance.
        gams_path (str, optional): Path to the GAMS solver. If None, GAMS will not be used.
        solver_name (str, optional): The solver to use ('highs', 'gams', or 'ipopt'). Defaults to 'highs' unless gams_path is provided.
        options (list, optional): Additional options for the solver.

    Returns:
        tuple: Results of the optimization and the updated model instance.
    """
    results = None

    # Use GAMS if gams_path is specified
    if gams_path and (solver_name is None or solver_name.lower() == 'gams'):
        pyo.pyomo.common.Executable('gams').set_path(gams_path)
        solver = pyo.SolverFactory('gams')
        print('GAMS solvers library availability:', solver.available())
        print('Solver path:', solver.executable())

        io_options = {
            'mtype': 'lp',  # Type of problem (lp, nlp, mip, minlp)
            'solver': 'CPLEX',  # Name of solver
        }

        if options is None:
            options = [
                'option optcr = 1e-15;',
                'option reslim = 3600;',
                'GAMS_MODEL.optfile = 1;',
                '$onecho > cplex.opt',
                'workmem=4096',
                'scaind=1',
                #'numericalemphasis=1',
                #'epmrk=0.99',
                #'eprhs=1E-9',
                '$offecho',
            ]

        results = solver.solve(
            model_instance,
            keepfiles=True,
            symbolic_solver_labels=True,
            tee=False,
            report_timing=False,
            io_options=io_options,
            add_options=options
        )

        model_instance.solutions.load_from(results)

    # Use IPOPT if explicitly specified
    elif solver_name and solver_name.lower() == 'ipopt':
        opt = pyo.SolverFactory('ipopt')
        if options:
            for option in options:
                opt.options[option] = True
        results = opt.solve(model_instance)

    # Default to HiGHS if no solver specified or if solver_name is 'highs'
    else:
        opt = appsi.solvers.Highs()
        results = opt.solve(model_instance)

    return results, model_instance


    print('Optimization problem solved')
    ## TODO: Add a check for infeasibility and other solver errors

    return results, model_instance