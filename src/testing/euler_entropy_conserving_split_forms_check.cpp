#include "euler_entropy_conserving_split_forms_check.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nspecies, int nstate>
EulerSplitEntropyCheck<dim, nspecies, nstate>::EulerSplitEntropyCheck(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nspecies, int nstate>
int EulerSplitEntropyCheck<dim, nspecies, nstate>::run_test() const
{
    int testfail = 0;
    const unsigned int n_fluxes = 4;

    using TwoPtFluxEnum = Parameters::AllParameters::TwoPointNumericalFlux;
    const std::array<TwoPtFluxEnum, n_fluxes> two_point_fluxes{{TwoPtFluxEnum::IR, TwoPtFluxEnum::CH, TwoPtFluxEnum::Ra, TwoPtFluxEnum::KG}};
    std::array<double, n_fluxes> tols{{5E-15, 5E-15, 5E-15, 1E-10}};
    std::array<double, n_fluxes> volume_term_tols{{2E-7, 2E-5, 5E-14, 5E-14}};
    if (nspecies > 1)
        tols = {{2E-6,2E-6,2E-6,2E-6}};
    const std::array<std::string, n_fluxes> flux_names{{"Ismail-Roe", "Chandrashekar", "Ranocha", "Kennedy-Gruber"}};

    for (unsigned int i = 0; i < n_fluxes; ++i){
        pcout << "-----------------------------------------------------------------------" << std::endl;
        pcout << "   Using " << flux_names[i] << " two-point flux" << std::endl;
        pcout << "-----------------------------------------------------------------------" << std::endl;
        
        const TwoPtFluxEnum flux = two_point_fluxes[i];
        const double tol = tols[i];
        const double volume_term_tol = volume_term_tols[i];

        // Copying parameters and modifying flux type
        PHiLiP::Parameters::AllParameters parameters = *(this->all_parameters);
        parameters.two_point_num_flux_type = flux;

        // Initialize flow_solver
        std::unique_ptr<FlowSolver::FlowSolver<dim,nspecies,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nspecies,nstate>::select_flow_case(&parameters, parameter_handler);

        // Compute  initial and final entropy
        flow_solver->flow_solver_case->compute_and_update_integrated_quantities(*flow_solver->dg);
        const double initial_KE = flow_solver->flow_solver_case->get_integrated_kinetic_energy();
        const double initial_entropy = flow_solver-> flow_solver_case->get_numerical_entropy(flow_solver->dg);
        double initial_volume_term = 0.0;
        if (nspecies > 1)
            initial_volume_term = flow_solver-> flow_solver_case->get_volume_term();

        static_cast<void>(flow_solver->run());
        flow_solver->flow_solver_case->compute_and_update_integrated_quantities(*flow_solver->dg);
        const double final_entropy = flow_solver->flow_solver_case->get_numerical_entropy(flow_solver->dg); 
        const double final_KE = flow_solver->flow_solver_case->get_integrated_kinetic_energy();
        double final_volume_term = 0.0;
        if (nspecies > 1)
            final_volume_term = flow_solver-> flow_solver_case->get_volume_term();

        //Compare initial and final entropy to confirm entropy preservation
        if (nspecies == 1)
            pcout << "Final numerical entropy: " << final_entropy << std::endl;
        else
            pcout << "Change in numerical entropy: " << abs(final_entropy-initial_entropy) << std::endl;
        pcout << "Initial kinetic energy: " << std::setprecision(16) << initial_KE << std::endl
              << "Final:                  " << final_KE << std::endl
              << "Scaled difference:      " << abs((initial_KE-final_KE)/initial_KE) << std::endl;
        if (nspecies > 1) {
            pcout  << "Initial volume work:    " << initial_volume_term << std::endl 
                   << "Final volume work:      " << final_volume_term << std::endl
                   << "Difference:             " << abs((initial_volume_term-final_volume_term));
        }
        pcout  << std::endl << std::endl;

        if (nspecies==1 && abs(final_entropy) > tol){
            pcout << "Entropy change is not within allowable tolerance. Test failing." << std::endl;
            testfail = 1;
        } else if (nspecies > 1 && abs(final_entropy-initial_entropy) > tol){
            pcout << "Entropy change is not within allowable tolerance. Test failing." << std::endl;
            testfail = 1;
        } else pcout << "Entropy change is allowable." << std::endl;

        if (nspecies > 1 && abs(final_volume_term - initial_volume_term) > volume_term_tol) {
            pcout << "Volume term change is not within allowable tolerance. Test failing." << std::endl;
            testfail = 1;
        }
    }

    return testfail;
}

#if PHILIP_DIM==3
    template class EulerSplitEntropyCheck<PHILIP_DIM, PHILIP_SPECIES,PHILIP_DIM+PHILIP_SPECIES+1>;
#endif
} // Tests namespace
} // PHiLiP namespace
