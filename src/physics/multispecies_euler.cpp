#include <cmath>
#include <vector>
#include <fstream>
#include <boost/preprocessor/seq/for_each.hpp>

#include "ADTypes.hpp"

#include "physics.h"
#include "euler.h"
#include "multispecies_euler.h" 

namespace PHiLiP {
namespace Physics {

template <int dim, int nspecies, int nstate, typename real>
MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>::MultiSpecies_CaloricallyPerfect_Euler ( 
    const Parameters::AllParameters *const                    parameters_input,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    std::shared_ptr< ManufacturedSolutionFunction<dim,nspecies,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type_input,
    const bool                                                has_nonzero_diffusion,
    const bool                                                has_nonzero_physical_source)
    : PhysicsBase<dim,nspecies,nstate,real>(parameters_input, has_nonzero_diffusion,has_nonzero_physical_source,manufactured_solution_function)
    , gam_ref(gamma_gas)
    , mach_ref(mach_inf)
    , mach_ref_sqr(mach_inf*mach_inf)
    , two_point_num_flux_type(two_point_num_flux_type_input)
    , Ru(8.31446261815324) /// [J/(mol·K)]
    , MW_Air(28.9651159 * pow(10,-3)) /// [kg/mol]
    , R_ref(Ru/MW_Air) /// = Ru/MW_Air [J/(kg·K)]
    , temperature_ref(298.15) /// [K]
    , u_ref(mach_ref*sqrt(gam_ref*R_ref*temperature_ref)) /// [m/s]
    , u_ref_sqr(u_ref*u_ref) /// [m/s]^2
    , density_ref(1.225) /// [kg/m^3]
{
    static_assert(nstate==dim+nspecies+1, "Physics::MultispeciesCaloricallyPerfect() should be created with nstate=PHILIP_DIM+PHILIP_SPECIES+1"); // Note: update this with nspecies in the future
    if(parameters_input->chemistry_input_file=="") {
        this->pcout << "Name of chemistry file containing NASA CAP data for species has not been passed in. Aborting..." << std::endl;
        std::abort(); 
    }
    readspeciesdata(parameters_input->chemistry_input_file);
}

// Read chemistry file
template <int dim, int nspecies, int nstate, typename real>
void MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::readspeciesdata(std::string NASADataFilename)
{
    std::string line, dum_char;

    std::ifstream chemfile (NASADataFilename);
    std::getline(chemfile, line);
    std::getline(chemfile, line);
    int N_species = (int)std::stof(line);
    if(nspecies != N_species) {
        std::cout << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl
                  << "Number of species in chemistry file does not match PHILIP_SPECIES." << std::endl
                  << "Number of species in file = " << N_species << " and PHILIP_SPECIES = " << PHILIP_SPECIES << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
    }

    std::string dummy_name;
    std::string::size_type sz1;
    //===============================================
    /*-------------------------------------------
     *           SPECIES SECTION
     *-------------------------------------------*/
    for(int i=0; i<nspecies; i++)
    {
        // Init
        sz1 = 0;
        std::getline(chemfile, line);
        std::getline(chemfile, line);
        std::getline(chemfile, line);
        species_name[i] = line;
        std::getline(chemfile, line);
        std::getline(chemfile, line);
        std::getline(chemfile, line);
        species_weight[i] = std::stof(line); // Species molecular weight [g/mol]
        species_weight[i] /= 1000.0; // Species molecular weight [kg/mol]

        std::getline(chemfile, line);
        std::getline(chemfile, line);
        std::getline(chemfile, line);
        species_Cp[i] = std::stof(line); // Species molecular weight [kJ/(kg·K)]
        species_Cp[i] *= 1000.0; // Species molecular weight [J/(kg·K)]
        species_Cp[i] /= this->R_ref; // nondimensionalized mass value

        std::getline(chemfile, line);
        std::getline(chemfile, line);
        std::getline(chemfile, line);
        species_enthalpy_offset[i] = std::stof(line); // Species enthalpy from T = 0 to T= 1 (nondimensional)

        std::getline(chemfile, line);
        std::getline(chemfile, line);
        std::getline(chemfile, line);
        species_entropy_offset[i] = std::stof(line); // Species entropy from T = 0 to T= 1 (nondimensional)

        std::getline(chemfile, line);
        std::getline(chemfile, line);
        std::getline(chemfile, line);
        for(int j=0; j<2; j++)
        {
            line = line.substr(sz1);
            sz1 = 0;
            NASACAPTemperatureLimits[i][j] = std::stof(line,&sz1);
        }

        std::getline(chemfile, line);
        std::getline(chemfile, line);
        // Init
        sz1 = 0;
        std::getline(chemfile, line);
        for(int j=0; j<6; j++) 
        {
            line = line.substr(sz1);
            sz1 = 0;
            Cp_poly_coeffs[i][j] = std::stod(line,&sz1);
        }
    }

    this->Rs = compute_Rs();
    for(int i=0; i<nspecies; i++) {
        this->species_Cv[i] = this->species_Cp[i] - this->Rs[i];
    }
}

template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    std::array<real,nstate> eig;
    real vel_dot_n = 0.0;
    for (int d=0;d<dim;++d) { vel_dot_n += vel[d]*normal[d]; };
    for (int i=0; i<nstate; i++) {
        eig[i] = vel_dot_n;
    }

    return eig;
}

template <int dim, int nspecies, int nstate, typename real>
real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &conservative_soln) const
{
    const real sound = compute_sound(conservative_soln);
    real vel2 = compute_velocity_squared_from_conservative_solution(conservative_soln);

    const real max_eig = sqrt(vel2) + sound;

    return max_eig;
}

template <int dim, int nspecies, int nstate, typename real>
real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::max_convective_normal_eigenvalue (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);

    const real sound = compute_sound (conservative_soln);

    real vel_dot_n = 0.0;
    for (int d=0;d<dim;++d) { vel_dot_n += vel[d]*normal[d]; };
    const real max_normal_eig = abs(vel_dot_n) + sound;

    return max_normal_eig;
}

template <int dim, int nspecies, int nstate, typename real>
real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::max_viscous_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    // zero because inviscid
    const real max_eig = 0.0;
    return max_eig;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
     std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    // No dissipative flux (i.e. viscous terms) for this physics class
    for (int i=0; i<nstate; i++) {
        diss_flux[i] = 0;
    }
    return diss_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::source_term (
    const dealii::Point<dim,real> &/*pos*/,
    const std::array<real,nstate> &/*conservative_soln*/,
    const real /*current_time*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    this->pcout<<"Source Terms not implemented for MultiSpecies_CaloricallyPerfect_Euler."<<std::endl;
    std::abort();
    std::array<real,nstate> source_term;
    source_term.fill(0.0);
    return source_term;
}

template <int dim, int nspecies, int nstate, typename real>
void MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::boundary_wall (
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // Slip wall boundary for Euler
    boundary_slip_wall(normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
}

template <int dim, int nspecies, int nstate, typename real>
void MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::boundary_slip_wall (
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    // Slip wall boundary conditions (No penetration)
    // Given by Algorithm II of the following paper
    // Krivodonova, L., and Berger, M.,
    // “High-order accurate implementation of solid wall boundary conditions in curved geometries,”
    // Journal of Computational Physics, vol. 211, 2006, pp. 492–512.
    const std::array<real,nstate> primitive_interior_values = convert_conservative_to_primitive(soln_int);

    // Copy density and pressure and mass fractions
    std::array<real,nstate> primitive_boundary_values;
    primitive_boundary_values[0] = primitive_interior_values[0];
    primitive_boundary_values[dim+1] = primitive_interior_values[dim+1];
    for (int ispecies = 0; ispecies < nspecies-1; ++ispecies) {
        primitive_boundary_values[dim+2+ispecies] = primitive_interior_values[dim+2+ispecies];
    }

    const dealii::Tensor<1,dim,real> surface_normal = -normal_int;
    dealii::Tensor<1,dim,real> velocities_int;
    for (int d=0; d<dim; d++) { velocities_int[d] = primitive_interior_values[1+d]; }
    //const dealii::Tensor<1,dim,real> velocities_bc = velocities_int - 2.0*(velocities_int*surface_normal)*surface_normal;
    real vel_int_dot_normal = 0.0;
    for (int d=0; d<dim; d++) {
        vel_int_dot_normal = vel_int_dot_normal + velocities_int[d]*surface_normal[d];
    }
    dealii::Tensor<1,dim,real> velocities_bc;
    for (int d=0; d<dim; d++) {
        velocities_bc[d] = velocities_int[d] - 2.0*(vel_int_dot_normal)*surface_normal[d];
        //velocities_bc[d] = velocities_int[d] - (vel_int_dot_normal)*surface_normal[d];
        //velocities_bc[d] += velocities_int[d] * surface_normal.norm_square();
    }
    for (int d=0; d<dim; ++d) {
        primitive_boundary_values[1+d] = velocities_bc[d];
    }

    const std::array<real,nstate> modified_conservative_boundary_values = convert_primitive_to_conservative(primitive_boundary_values);
    for (int istate=0; istate<nstate; ++istate) {
        soln_bc[istate] = modified_conservative_boundary_values[istate];
    }

    for (int istate=0; istate<nstate; ++istate) {
        soln_grad_bc[istate] = -soln_grad_int[istate];
    }
}

template <int dim, int nspecies, int nstate, typename real>
void MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, real> &/*pos*/,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    if (boundary_type == 1001) {
        // Wall boundary condition (slip for Real Gas, no-slip for Navier-Stokes-Real-Gas)
        boundary_wall (normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } else if (boundary_type == 1006) {
        // Slip wall boundary condition
        boundary_slip_wall (normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } else {
        this->pcout<<"Boundary condition #" << boundary_type << " not implemented for MultiSpecies_CaloricallyPerfect_Euler."<<std::endl;
        std::abort();
    }
}

// Details of the following algorithms are presented in Liki's Master's thesis.
/* MAIN FUNCTIONS */
// Algorithm 1 (f_M1): Compute mixture density
template <int dim, int nspecies, int nstate, typename real>
template<typename real2>
inline real2 MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
:: compute_mixture_density ( const std::array<real2,nstate> &conservative_soln ) const
{
    const real2 mixture_density = conservative_soln[0];

    return mixture_density;
}

// Algorithm 2 (f_M2): Compute velocities
template <int dim, int nspecies, int nstate, typename real>
inline dealii::Tensor<1,dim,real> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_velocities ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    dealii::Tensor<1,dim,real> vel;
    for (int d=0; d<dim; ++d) { vel[d] = conservative_soln[1+d]/mixture_density; }

    return vel;
}

// Algorithm 3 (f_M3): Compute squared velocities
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_velocity_squared_from_conservative_solution ( const std::array<real,nstate> &conservative_soln ) const
{
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    real vel2 = 0.0;
    for (int d=0; d<dim; d++) { 
        vel2 = vel2 + vel[d]*vel[d]; 
    }  

    return vel2;
}

template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_velocity_squared ( const dealii::Tensor<1,dim,real> &velocities ) const
{
    real vel2 = 0.0;
    for (int d=0; d<dim; d++) { 
        vel2 = vel2 + velocities[d]*velocities[d]; 
    }  

    return vel2;
}

template <int dim, int nspecies, int nstate, typename real>
inline dealii::Tensor<1,dim,real> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::extract_velocities_from_primitive ( const std::array<real,nstate> &primitive_soln ) const
{
    dealii::Tensor<1,dim,real> velocities;
    for (int d=0; d<dim; d++) { velocities[d] = primitive_soln[1+d]; }
    return velocities;
}

// Algorithm 4 (f_M4): Compute specific kinetic energy
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_specific_kinetic_energy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real vel2 = compute_velocity_squared_from_conservative_solution(conservative_soln);
    const real k = 0.5*vel2;

    return k;
}

// Algorithm 5 (f_M5): Compute mixture specific total energy
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_mixture_specific_total_energy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    const real mixture_specific_total_energy = conservative_soln[dim+1]/mixture_density;

    return mixture_specific_total_energy;
}

// Algorithm 6 (f_M6): Compute species densities
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_densities ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    std::array<real,nspecies> species_densities;
    real sum = 0.0;
    for (int s=0; s<nspecies-1; ++s) 
    { 
        species_densities[s] = conservative_soln[dim+2+s]; 
        sum += species_densities[s];
    }
    species_densities[nspecies-1] = mixture_density - sum;

    return species_densities;
}

// Algorithm 7 (f_M7): Compute mass fractions
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_mass_fractions ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    const std::array<real,nspecies> species_densities = compute_species_densities(conservative_soln);
    std::array<real,nspecies> mass_fractions;
    for (int s=0; s<nspecies; ++s) 
    { 
        mass_fractions[s] = species_densities[s]/mixture_density; 
    }

    return mass_fractions;
}

// Algorithm 8 (f_M8): Compute mixture from species
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_mixture_from_species ( const std::array<real,nspecies> &mass_fractions, const std::array<real,nspecies> &species) const
{
    real mixture = 0.0; 
    for (int s=0; s<nspecies; ++s) 
    { 
        mixture += mass_fractions[s]*species[s]; 
    }   

    return mixture;
}

// Algorithm 9 (f_M9): Compute dimensional temperature
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_dimensional_temperature ( const real temperature ) const
{
    const real dimensional_temperature = temperature*this->temperature_ref;

    return dimensional_temperature;
}

// Algorithm 10 (f_M10): Compute species gas constants
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_Rs () const
{
    std::array<real,nspecies> Rs;
    for (int s=0; s<nspecies; ++s) 
    {
        Rs[s] = Ru/this->species_weight[s]/this->R_ref;
    }
    return Rs;
}

// Algorithm 13 (f_M13): Compute species specific enthalpy
// This function has been modified by Shruthi
// Modification: separates the temperature index into its own separate function since two different functions use it
// Modification #2: includes a clipping process to ensure we can still calculate for temps outside range
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_specific_enthalpy ( const real temperature ) const
{
    std::array<real,nspecies> e = compute_species_specific_internal_energy(temperature);
    std::array<real,nspecies> h;
    
    for (int s=0; s<nspecies; ++s) 
    {
        h[s] = e[s]*(this->u_ref_sqr/(this->R_ref*this->temperature_ref)) + this->Rs[s]*temperature;
    }
    return h;
}

// Algorithm 14 (f_M14): Compute species specific internal energy
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_specific_internal_energy( const real temperature) const
{
    std::array<real,nspecies> e;
    for (int s=0; s<nspecies; ++s) 
    {
        e[s] = (this->species_Cv[s]*temperature)*((this->R_ref*this->temperature_ref)/u_ref_sqr);
    }
    return e;
}

template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_mixture_internal_energy( const std::array<real,nstate> &conservative_soln ) const
{
    const real E = this->compute_mixture_specific_total_energy(conservative_soln);
    const real k = this->compute_specific_kinetic_energy(conservative_soln);

    const real e = E-k;

    return e;
}

template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_internal_energy( const std::array<real,nstate> &conservative_soln ) const
{
    return compute_mixture_internal_energy(conservative_soln);
}


// Compute species entropy by calculating integral and adding in density contribution (ie. R_k ln \rho_k)
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_species_entropy (
    const std::array<real,nstate> &conservative_soln) const
{
    const real temperature = compute_temperature(conservative_soln);
    const std::array<real,nspecies> species_densities = compute_species_densities(conservative_soln);
    std::array<real,nspecies> species_entropy;

    for(int ispecies = 0; ispecies < nspecies; ispecies++) {
        species_entropy[ispecies] = this->species_Cv[ispecies]*log(temperature*this->temperature_ref) - this->Rs[ispecies]*log(species_densities[ispecies]*this->density_ref);
    }
    return species_entropy;
}


// Compute mixture entropy
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_entropy (
    const std::array<real,nstate> &conservative_soln) const
{
    const std::array<real,nspecies> species_entropy = compute_species_entropy(conservative_soln);
    const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);

    const real entropy = compute_mixture_from_species(mass_fractions,species_entropy);
    if(entropy != entropy) {
        std::cout << "The calculated entropy is NaN - this is likely due to a species having a mass fraction of zero...Aborting." << std::endl;
        std::abort();
    }

    return entropy;
}

template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_numerical_entropy_function ( const std::array<real,nstate> &conservative_soln ) const
{
    const real density = conservative_soln[0];

    const real entropy = compute_entropy(conservative_soln);

    const real numerical_entropy_function = - density * entropy;

    return numerical_entropy_function;
}

// Compute Gibbs' energy of species using species entropy and species Cp
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_species_gibbs_energy (
    const std::array<real,nstate> &conservative_soln) const
{
    const real temperature = compute_temperature(conservative_soln);

    std::array<real,nspecies> species_entropy = compute_species_entropy(conservative_soln);

    std::array<real, nspecies> species_gibbs;
    for(int ispecies = 0; ispecies < nspecies; ++ispecies) {
        species_gibbs[ispecies] = temperature*(this->species_Cp[ispecies] - species_entropy[ispecies]);
    }

    return species_gibbs;
}

// Compute the entropy variables from conservative solution
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_entropy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<real,nstate> entropy_var;
    const real temperature = compute_temperature(conservative_soln);
    std::array<real,nspecies> species_gibbs = compute_species_gibbs_energy(conservative_soln);
    real vel2 = compute_velocity_squared_from_conservative_solution(conservative_soln);

    entropy_var[0] = species_gibbs[nspecies-1] - (0.5*vel2);
    entropy_var[dim+1] = -1.0;

    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    for (int idim = 0; idim < dim; ++idim) {
        entropy_var[idim+1] = vel[idim];
    }

    for (int ispecies = 0; ispecies < nspecies - 1; ++ispecies) {
        entropy_var[dim+2+ispecies] = species_gibbs[ispecies] - species_gibbs[nspecies-1];
    }

    for (int istate = 0; istate < nstate; ++istate) {
        entropy_var[istate] /= temperature;
    }
    return entropy_var;
}

// Map entropy variables back to conservative solution
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_conservative_variables_from_entropy_variables (
    const std::array<real,nstate> &entropy_var) const
{
    std::array<real,nstate> conservative_var;
    const real temperature = -1/entropy_var[dim+1];
    const int nth_species_idx = nspecies - 1;

    std::array<real,nspecies> species_gibbs;

    real entropy_var_vel_squared = 0.0;
    for(int idim=0; idim<dim; idim++){
        entropy_var_vel_squared += pow(entropy_var[idim + 1]*temperature, 2.0);
    }

    species_gibbs[nth_species_idx] = temperature*entropy_var[0] + entropy_var_vel_squared/2.0;
    for(int ispecies = 0; ispecies < nth_species_idx; ++ispecies) {
        species_gibbs[ispecies] = temperature*entropy_var[dim+2+ispecies] + species_gibbs[nth_species_idx];
    }

    std::array<real,nspecies> species_entropy;
    for(int ispecies = 0; ispecies < nth_species_idx; ++ispecies) {
        species_entropy[ispecies] = this->species_Cp[ispecies] - (species_gibbs[ispecies]/temperature);
    }
    species_entropy[nth_species_idx] = species_Cp[nth_species_idx] - (species_gibbs[nth_species_idx]/temperature);

    std::array<real,nspecies> species_density;
    conservative_var[0] = 0.0;
    for(int ispecies = 0; ispecies < nspecies; ++ispecies) {
        species_density[ispecies] = (exp((1/this->Rs[ispecies])*(this->species_Cv[ispecies]*log(temperature*this->temperature_ref) - species_entropy[ispecies])))*(1.0/this->density_ref);
        conservative_var[0] += species_density[ispecies];

        if (dim + 2 + ispecies < nstate)
            conservative_var[dim+2+ispecies] = species_density[ispecies];
    }

    const real mixture_density = conservative_var[0];

    for (int idim = 0; idim < dim; ++idim) {
        conservative_var[idim+1] = mixture_density*entropy_var[idim+1]*temperature;
    }
    
    // specific kinetic energy
    const real specific_kinetic_energy = 0.50*entropy_var_vel_squared;
    // species specific enthalpy
    std::array<real,nspecies> species_specific_internal_energy;
    // species energy
    for (int s=0; s<nspecies; ++s) 
    { 
      species_specific_internal_energy[s] = (this->species_Cv[s]*temperature)*((this->R_ref*this->temperature_ref)/u_ref_sqr);
    }
    std::array<real,nspecies> mass_fractions;
    for (int ispecies = 0; ispecies < nspecies; ++ispecies) {
        mass_fractions[ispecies] = species_density[ispecies]/mixture_density;
    }
    const real mixture_internal_energy = compute_mixture_from_species(mass_fractions,species_specific_internal_energy);
    const real mixture_specific_total_energy = specific_kinetic_energy + mixture_internal_energy;
  
    conservative_var[dim+1] = mixture_density*mixture_specific_total_energy;

    return conservative_var;
}

// Computes the kinetic energy variables (Based off Cicchino 2025, Eq. 59)
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_kinetic_energy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<real,nstate> kin_energy_var;
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    const real vel2 = compute_velocity_squared_from_conservative_solution(conservative_soln);

    kin_energy_var[0] = - 0.5 * vel2;
    for(int idim=0; idim<dim; idim++){
        kin_energy_var[idim+1] = vel[idim];
    }
    kin_energy_var[dim+1] = 0.0;
    for(int ispecies=0; ispecies<nspecies-1; ispecies++) {
        int index = dim+2+ispecies;
        kin_energy_var[index] = 0.0;
    }

    return kin_energy_var;
}

// Algorithm 15 (f_M15): Compute temperature
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_temperature ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    const real mixture_gas_constant = compute_mixture_gas_constant(conservative_soln);
    const real mixture_pressure = compute_mixture_pressure(conservative_soln);
    const real temperature = (mixture_pressure/(mixture_density*mixture_gas_constant))*(this->u_ref_sqr/(this->R_ref*this->temperature_ref));

    return temperature;
}

// Algorithm 16 (f_M16): Compute mixture gas constant
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_mixture_gas_constant ( const std::array<real,nstate> &conservative_soln ) const
{
    const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);
    const real mixture_gas_constant = compute_mixture_from_species(mass_fractions,this->Rs);
    return mixture_gas_constant;
}

// Algorithm 17 (f_M17): Compute mixture pressure
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_mixture_pressure ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = compute_mixture_density(conservative_soln);
    const real mixture_gamma = compute_gamma(conservative_soln);
    const real E = this->compute_mixture_specific_total_energy(conservative_soln);
    const real k = this->compute_specific_kinetic_energy(conservative_soln);
    const real mixture_pressure = mixture_density*(mixture_gamma-1.0)*(E-k);

    return mixture_pressure;
}

// Algorithm 17 (f_M17): Compute pressure -> calls compute_pressure (allows other classes to use PhysicsBase ptr)
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_pressure ( const std::array<real,nstate> &conservative_soln ) const
{
    return compute_mixture_pressure(conservative_soln);
}

template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_pressure_from_density_temperature ( const real density, const real temperature, const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_gas_constant = compute_mixture_gas_constant(conservative_soln);
    const real mixture_pressure = density*mixture_gas_constant*temperature/(this->gam_ref*this->mach_ref_sqr);
    return mixture_pressure;
}

// Algorithm 18 (f_M18): Compute mixture specific total enthalpy
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_mixture_specific_total_enthalpy ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_specific_total_energy = compute_mixture_specific_total_energy(conservative_soln);
    const real mixture_pressure = compute_mixture_pressure(conservative_soln);
    const real mixture_density = compute_mixture_density(conservative_soln);
    const real mixture_specific_total_enthalpy = (mixture_specific_total_energy + mixture_pressure/mixture_density)*(this->u_ref_sqr/(this->R_ref*this->temperature_ref));

    return mixture_specific_total_enthalpy;
}

// Algorithm 19 (f_M19): Compute convective flux
template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const  
{
    /* definitions */
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const real mixture_density = compute_mixture_density(conservative_soln);
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    const real mixture_pressure = compute_mixture_pressure(conservative_soln);
    const std::array<real,nspecies> species_densities = compute_species_densities(conservative_soln);
    const real mixture_specific_total_energy = compute_mixture_specific_total_energy(conservative_soln);

    // flux dimension loop; E -> F -> G
    for (int flux_dim=0; flux_dim<dim; ++flux_dim) 
    {
        /* A) mixture density equations */
        conv_flux[0][flux_dim] = conservative_soln[1+flux_dim];

        /* B) mixture momentum equations */
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim)
        {
            conv_flux[1+velocity_dim][flux_dim] = mixture_density*vel[flux_dim]*vel[velocity_dim];
        }
        conv_flux[1+flux_dim][flux_dim] += mixture_pressure; // Add diagonal of pressure

        /* C) mixture energy equations */
        conv_flux[dim+1][flux_dim] = mixture_density*vel[flux_dim]*(mixture_specific_total_energy + mixture_pressure/mixture_density);

        /* D) species density equations */
        for (int s=0; s<nspecies-1; ++s)
        {
             conv_flux[dim+2+s][flux_dim] = species_densities[s]*vel[flux_dim];
        }
    }

    return conv_flux;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<2,nstate,real> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::convective_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    // Real Gas version of function in Euler
    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    real vel_normal = 0.0;
    for (int d=0;d<dim;d++) { vel_normal += vel[d] * normal[d]; }

    const real gam = compute_gamma(conservative_soln);
    const real gamm1 = gam - 1.0;
    const real vel2 = compute_velocity_squared_from_conservative_solution(conservative_soln);
    const real phi = 0.5*gamm1 * vel2;

    const real density = conservative_soln[0];
    const real tot_energy = conservative_soln[dim+1];
    const real E = tot_energy / density;
    const real a1 = gam*E-phi;
    const real a2 = gamm1;
    const real a3 = gam-2.0;

    dealii::Tensor<2,nstate,real> jacobian;
    for (int d=0; d<dim; ++d) {
        jacobian[0][1+d] = normal[d];
    }
    for (int row_dim=0; row_dim<dim; ++row_dim) {
        jacobian[1+row_dim][0] = normal[row_dim]*phi - vel[row_dim] * vel_normal;
        for (int col_dim=0; col_dim<dim; ++col_dim){
            if (row_dim == col_dim) {
                jacobian[1+row_dim][1+col_dim] = vel_normal - a3*normal[row_dim]*vel[row_dim];
            } else {
                jacobian[1+row_dim][1+col_dim] = normal[col_dim]*vel[row_dim] - a2*normal[row_dim]*vel[col_dim];
            }
        }
        jacobian[1+row_dim][dim+1] = normal[row_dim]*a2;
    }
    jacobian[dim+1][0] = vel_normal*(phi-a1);
    for (int d=0; d<dim; ++d){
        jacobian[dim+1][1+d] = normal[d]*a1 - a2*vel[d]*vel_normal;
    }
    jacobian[dim+1][dim+1] = gam*vel_normal;

    return jacobian;
}

// Helper function to compute mean for split fluxes
template <int dim, int nspecies, int nstate, typename real>
real MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_average(const real val1, const real val2) const
{
    // Compute mean of the two values passed in
    const real mean_val = (val1+val2)/(2.0);

    return mean_val;
}

// Helper function to compute logarithmic mean for split fluxes
template <int dim, int nspecies, int nstate, typename real>
real MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_ismail_roe_logarithmic_mean(const real val1, const real val2) const
{
    // See Appendix B [Ismail and Roe, 2009, Entropy-Consistent Euler Flux Functions II]
    // -- Numerically stable algorithm for computing the logarithmic mean
    if(val1 < 1e-16 || val2 < 1e-16)
        return 0;

    const real zeta = val1/val2;
    const real f = (zeta-1.0)/(zeta+1.0);
    const real u = f*f;

    real F;
    if(u<1.0e-2){ F = 1.0 + u/3.0 + u*u/5.0 + u*u*u/7.0; } 
    else { 
        if constexpr(std::is_same<real,double>::value) F = std::log(zeta)/2.0/f; 
    }

    const real log_mean_val = (val1+val2)/(2.0*F);

    return log_mean_val;
}

///  Evaluates convective flux based on the chosen split form.
template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::convective_numerical_split_flux(const std::array<real,nstate> &conservative_soln1,
                                  const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    if(two_point_num_flux_type == two_point_num_flux_enum::KG) {
        conv_num_split_flux = convective_numerical_split_flux_kennedy_gruber(conservative_soln1, conservative_soln2);
    } else if(two_point_num_flux_type == two_point_num_flux_enum::IR) {
        conv_num_split_flux = convective_numerical_split_flux_ismail_roe(conservative_soln1, conservative_soln2);
    } else if(two_point_num_flux_type == two_point_num_flux_enum::CH) {
        conv_num_split_flux = convective_numerical_split_flux_chandrashekar(conservative_soln1, conservative_soln2);
    } else if(two_point_num_flux_type == two_point_num_flux_enum::Ra) {
                conv_num_split_flux = convective_numerical_split_flux_ranocha(conservative_soln1, conservative_soln2);
    }

    return conv_num_split_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::convective_numerical_split_flux_kennedy_gruber(const std::array<real,nstate> &conservative_soln1,
                                                 const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    const std::array<real,nspecies> rho_species1 = compute_species_densities(conservative_soln1);
    const std::array<real,nspecies> rho_species2 = compute_species_densities(conservative_soln2);

    // compute mean densities
    std::array<real, nspecies> mean_species_densities;
    real mean_density = 0.0;
    for (int ispecies = 0; ispecies < nspecies; ++ispecies) {
        mean_species_densities[ispecies] = (rho_species1[ispecies]+rho_species2[ispecies])/2.0;
        mean_density += mean_species_densities[ispecies];
    }

    // compute mean velocities
    dealii::Tensor<1,dim,real> vel_1 = compute_velocities(conservative_soln1);
    dealii::Tensor<1,dim,real> vel_2 = compute_velocities(conservative_soln2);
    dealii::Tensor<1,dim,real> mean_vel;
    for (int d=0; d<dim; ++d) {
        mean_vel[d] = 0.5*(vel_1[d]+vel_2[d]);
    }

    // compute mean pressure
    real pressure1 = compute_mixture_pressure(conservative_soln1);
    real pressure2 = compute_mixture_pressure(conservative_soln2);
    real mean_pressure = (pressure1 + pressure2)/2.0;

    // compute mean total energy
    real total_energy1 = compute_mixture_specific_total_energy(conservative_soln1);
    real total_energy2 = compute_mixture_specific_total_energy(conservative_soln2);
    real mean_total_energy = (total_energy1 + total_energy2)/2.0;

    for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    {
        // Density equation
        conv_num_split_flux[0][flux_dim] = mean_density * mean_vel[flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = mean_density*mean_vel[flux_dim]*mean_vel[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += mean_pressure; // Add diagonal of pressure
        // Energy equation
        conv_num_split_flux[dim+1][flux_dim] = mean_density*mean_vel[flux_dim]*mean_total_energy + mean_pressure * mean_vel[flux_dim];
        // Species density equation
        for (int ispecies = 0; ispecies < nspecies - 1; ++ispecies) {
            conv_num_split_flux[dim+2+ispecies][flux_dim] = mean_species_densities[ispecies] * mean_vel[flux_dim];
        }
    }

    return conv_num_split_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::convective_numerical_split_flux_ismail_roe(const std::array<real,nstate> &conservative_soln1,
                                                 const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;

    // compute the sqrt of TL, TR, 1/TL, 1/TR
    const real sqrt_T1 = sqrt(compute_temperature(conservative_soln1));
    const real sqrt_T2 = sqrt(compute_temperature(conservative_soln2));
    const real inv_sqrt_T1 = 1.0/sqrt_T1;
    const real inv_sqrt_T2 = 1.0/sqrt_T2;
    const real avg_inv_sqrt_T = compute_average(inv_sqrt_T1,inv_sqrt_T2);
    const real log_mean_inv_sqrt_T = compute_ismail_roe_logarithmic_mean(inv_sqrt_T1,inv_sqrt_T2);

    // compute all different avg/mean densities required for flux
    std::array<real,nspecies> species_densities1 = compute_species_densities(conservative_soln1);
    std::array<real,nspecies> species_densities2 = compute_species_densities(conservative_soln2);
    std::array<real,nspecies> log_mean_species_densities_sqrt_temp;
    real sum_of_Rk_rhok = 0.0;
    real mean_density = 0.0;
    for (int ispecies = 0; ispecies < nspecies; ispecies++){
        log_mean_species_densities_sqrt_temp[ispecies] = compute_ismail_roe_logarithmic_mean(species_densities1[ispecies]*sqrt_T1, species_densities2[ispecies]*sqrt_T1);
        sum_of_Rk_rhok += this->Rs[ispecies]*compute_average(species_densities1[ispecies]*sqrt_T1, species_densities2[ispecies]*sqrt_T2);
        mean_density += log_mean_species_densities_sqrt_temp[ispecies];
    }

    // compute all avg/mean velocity terms required for the flux
    const dealii::Tensor<1,dim,real> vel1 = compute_velocities(conservative_soln1);
    const dealii::Tensor<1,dim,real> vel2 = compute_velocities(conservative_soln2);
    dealii::Tensor<1,dim,real> vel_avg;
    // real vel_sqr_avg = 0.0;
    // real vel_avg_sqr = 0.0;

    for (int d=0; d<dim; ++d) {
        vel_avg[d] = compute_average(vel1[d]*inv_sqrt_T1,vel2[d]*inv_sqrt_T2);
        // vel_sqr_avg += compute_average(vel1[d]*vel1[d],vel2[d]*vel2[d]);
        // vel_avg_sqr += vel_avg[d]*vel_avg[d];
    }

    for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    {
        // Density equation
        conv_num_split_flux[0][flux_dim] = mean_density * vel_avg[flux_dim];

        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = (mean_density*vel_avg[flux_dim]*vel_avg[velocity_dim])/avg_inv_sqrt_T;
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += (sum_of_Rk_rhok/avg_inv_sqrt_T)/(this->gam_ref*this->mach_ref_sqr); // Add diagonal of pressure

        // Species density equation
        for (int ispecies = 0; ispecies < nspecies - 1; ++ispecies) {
            conv_num_split_flux[dim+2+ispecies][flux_dim] = log_mean_species_densities_sqrt_temp[ispecies] * vel_avg[flux_dim];
        }

        // compute energy term sum
        real energy_sum_of_species_CvT = 0.0;
        for (int ispecies = 0; ispecies < nspecies; ++ispecies){
            energy_sum_of_species_CvT += (1.0/(avg_inv_sqrt_T*log_mean_inv_sqrt_T))*(this->species_Cv[ispecies]+0.5*this->Rs[ispecies])
                                            *((this->R_ref*this->temperature_ref)/u_ref_sqr)*log_mean_species_densities_sqrt_temp[ispecies]*vel_avg[flux_dim];
        }
        // compute pressure component of energy flux
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            energy_sum_of_species_CvT += (vel_avg[flux_dim]/(2.0*avg_inv_sqrt_T))*conv_num_split_flux[1+velocity_dim][flux_dim];
        }
        // Energy equation
        conv_num_split_flux[dim+1][flux_dim] = energy_sum_of_species_CvT;
    }
    // std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux_kg = convective_numerical_split_flux_kennedy_gruber(conservative_soln1,conservative_soln2);
    // for (int flux_dim = 0; flux_dim < dim; ++flux_dim){
    //     for (int istate = 0; istate < nstate; ++istate) {
    //         std::cout << " IR " << istate << " " << flux_dim << " " << conv_num_split_flux[istate][flux_dim] << std::endl;
    //         std::cout << " KG " << istate << " " << flux_dim << " " << conv_num_split_flux_kg[istate][flux_dim] << std::endl << std::endl;
    //     }
    // }
    // sleep(1);
    return conv_num_split_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::convective_numerical_split_flux_chandrashekar(const std::array<real,nstate> &conservative_soln1,
                                                 const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;

    // compute all different avg/mean densities required for flux
    std::array<real,nspecies> species_densities1 = compute_species_densities(conservative_soln1);
    std::array<real,nspecies> species_densities2 = compute_species_densities(conservative_soln2);
    std::array<real,nspecies> log_mean_species_densities;
    real sum_of_Rk_rhok = 0.0;
    real mean_density = 0.0;
    for (int ispecies = 0; ispecies < nspecies; ispecies++){
        log_mean_species_densities[ispecies] = compute_ismail_roe_logarithmic_mean(species_densities1[ispecies], species_densities2[ispecies]);
        sum_of_Rk_rhok += this->Rs[ispecies]*compute_average(species_densities1[ispecies], species_densities2[ispecies]);
        mean_density += log_mean_species_densities[ispecies];
    }

    // compute all avg/mean velocity terms required for the flux
    const dealii::Tensor<1,dim,real> vel1 = compute_velocities(conservative_soln1);
    const dealii::Tensor<1,dim,real> vel2 = compute_velocities(conservative_soln2);
    dealii::Tensor<1,dim,real> vel_avg;
    real vel_sqr_avg = 0.0;

    for (int d=0; d<dim; ++d) {
        vel_avg[d] = compute_average(vel1[d],vel2[d]);
        vel_sqr_avg += compute_average(vel1[d]*vel1[d],vel2[d]*vel2[d]);
    }

    // compute all avg/mean temperature terms required for the flux
    const real temperature1 = compute_temperature(conservative_soln1);
    const real temperature2 = compute_temperature(conservative_soln2);
    const real avg_inv_temp = compute_average((1.0/temperature1),(1.0/temperature2));
    const real log_mean_inv_temp = compute_ismail_roe_logarithmic_mean((1.0/temperature1),(1.0/temperature2));

    for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    {
        // Density equation
        conv_num_split_flux[0][flux_dim] = mean_density * vel_avg[flux_dim];

        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = mean_density*vel_avg[flux_dim]*vel_avg[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += (sum_of_Rk_rhok/avg_inv_temp)/(this->gam_ref*this->mach_ref_sqr); // Add diagonal of pressure

        // initialize sum of internal energy for total energy flux
        real energy_sum_of_species_CvT = 0.0;

        // Species density equation
        for (int ispecies = 0; ispecies < nspecies - 1; ++ispecies) {
            conv_num_split_flux[dim+2+ispecies][flux_dim] = log_mean_species_densities[ispecies] * vel_avg[flux_dim];
            energy_sum_of_species_CvT += ((this->species_Cv[ispecies]/log_mean_inv_temp)*((this->R_ref*this->temperature_ref)/u_ref_sqr) 
                                            - 0.5*vel_sqr_avg)*conv_num_split_flux[dim+2+ispecies][flux_dim];
        }
        // add contribution from last species which doesn't have a flux associated with it
        energy_sum_of_species_CvT += ((this->species_Cv[nspecies-1]/log_mean_inv_temp)*((this->R_ref*this->temperature_ref)/u_ref_sqr) 
                                            - 0.5*vel_sqr_avg)*log_mean_species_densities[nspecies-1]*vel_avg[flux_dim];

        // Energy equation
        conv_num_split_flux[dim+1][flux_dim] = energy_sum_of_species_CvT;
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[dim+1][flux_dim] += vel_avg[flux_dim]*conv_num_split_flux[1+velocity_dim][flux_dim];
        }
    }

    return conv_num_split_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim, nspecies, nstate, real>
::convective_numerical_split_flux_ranocha(const std::array<real,nstate> &conservative_soln1,
                                                 const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;

    // compute all different avg/mean densities required for flux
    std::array<real,nspecies> species_densities1 = compute_species_densities(conservative_soln1);
    std::array<real,nspecies> species_densities2 = compute_species_densities(conservative_soln2);
    std::array<real,nspecies> log_mean_species_densities;
    real sum_of_Rk_rhok = 0.0;
    real mean_density = 0.0;
    for (int ispecies = 0; ispecies < nspecies; ispecies++){
        log_mean_species_densities[ispecies] = compute_ismail_roe_logarithmic_mean(species_densities1[ispecies], species_densities2[ispecies]);
        sum_of_Rk_rhok += this->Rs[ispecies]*compute_average(species_densities1[ispecies], species_densities2[ispecies]);
        mean_density += log_mean_species_densities[ispecies];
    }

    // compute avg pressure term required for the flux
    const real pressure1 = compute_mixture_pressure(conservative_soln1);
    const real pressure2 = compute_mixture_pressure(conservative_soln2);
    const real avg_pressure = compute_average(pressure1, pressure2);

    // compute all avg/mean velocity terms required for the flux
    const dealii::Tensor<1,dim,real> vel1 = compute_velocities(conservative_soln1);
    const dealii::Tensor<1,dim,real> vel2 = compute_velocities(conservative_soln2);
    dealii::Tensor<1,dim,real> vel_avg;
    real vel_sqr_avg = 0.0;

    for (int d=0; d<dim; ++d) {
        vel_avg[d] = compute_average(vel1[d],vel2[d]);
        vel_sqr_avg += compute_average(vel1[d]*vel1[d],vel2[d]*vel2[d]);
    }

    // compute all avg/mean temperature terms required for the flux
    const real temperature1 = compute_temperature(conservative_soln1);
    const real temperature2 = compute_temperature(conservative_soln2);
    const real log_mean_inv_temp = compute_ismail_roe_logarithmic_mean((1.0/temperature1),(1.0/temperature2));

    for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    {
        // Density equation
        conv_num_split_flux[0][flux_dim] = mean_density * vel_avg[flux_dim];

        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = mean_density*vel_avg[flux_dim]*vel_avg[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += avg_pressure;// Add diagonal of pressure

         // initialize sum of internal energy for total energy flux
        real energy_sum_of_species_CvT = 0.0;

        // Species density equation
        for (int ispecies = 0; ispecies < nspecies - 1; ++ispecies) {
            conv_num_split_flux[dim+2+ispecies][flux_dim] = log_mean_species_densities[ispecies] * vel_avg[flux_dim];
            energy_sum_of_species_CvT += ((this->species_Cv[ispecies]/log_mean_inv_temp)*((this->R_ref*this->temperature_ref)/u_ref_sqr) 
                                            - 0.5*vel_sqr_avg)*conv_num_split_flux[dim+2+ispecies][flux_dim];
        }
        // add contribution from last species which doesn't have a flux associated with it
        energy_sum_of_species_CvT += ((this->species_Cv[nspecies-1]/log_mean_inv_temp)*((this->R_ref*this->temperature_ref)/u_ref_sqr) 
                                            - 0.5*vel_sqr_avg)*log_mean_species_densities[nspecies-1]*vel_avg[flux_dim];

        conv_num_split_flux[dim+1][flux_dim] = energy_sum_of_species_CvT;
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[dim+1][flux_dim] += vel_avg[flux_dim]*conv_num_split_flux[1+velocity_dim][flux_dim];
        }

        // compute additional terms from pressure fix
        real pressure_fix = 0.25*(pressure1-pressure2)*(vel1[flux_dim]-vel2[flux_dim]);

        // Energy equation
        conv_num_split_flux[dim+1][flux_dim] -= pressure_fix;
    }

    return conv_num_split_flux;
}

/* Supporting FUNCTIONS */
// Algorithm 20 (f_S20): Convert primitive to conservative
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const 
{
    /* definitions */
    std::array<real, nstate> conservative_soln;
    const real mixture_density = compute_mixture_density(primitive_soln);
    std::array<real, dim> vel;

    real vel2 = 0.0;
    real sum = 0.0;
    std::array<real,nspecies> species_densities;
    std::array<real,nspecies> mass_fractions;
    const real mixture_pressure = primitive_soln[dim+1];

    /* mixture density */
    conservative_soln[0] = mixture_density;

    /* mixture momentum */
    for (int d=0; d<dim; ++d) 
    {
        vel[d] = primitive_soln[1+d];
        vel2 = vel2 + vel[d]*vel[d]; ;
        conservative_soln[1+d] = mixture_density*vel[d];
    }

    /* mixture energy */
    // mass fractions
    for (int s=0; s<nspecies-1; ++s) 
    { 
        mass_fractions[s] = primitive_soln[dim+2+s];
        sum += mass_fractions[s];
    }
    mass_fractions[nspecies-1] = 1.00 - sum;     
    // species densities
    for (int s=0; s<nspecies; ++s) 
    { 
        species_densities[s] = mixture_density*mass_fractions[s];
    }

    // mixture gas constant
    const real mixture_R = compute_mixture_from_species(mass_fractions,this->Rs);
    const real temperature = (mixture_pressure/(mixture_density*mixture_R))*(this->u_ref_sqr/(this->R_ref*this->temperature_ref));
    // mixture internal energy
    std::array<real,nspecies> species_internal_energy = compute_species_specific_internal_energy(temperature);
    const real mixture_internal_energy = compute_mixture_from_species(mass_fractions, species_internal_energy);
    // specific kinetic energy
    const real specific_kinetic_energy = 0.50*vel2;  
    // mixture energy
    const real mixture_specific_total_energy = specific_kinetic_energy + mixture_internal_energy;
    conservative_soln[dim+1] = mixture_density*mixture_specific_total_energy;

    /* species densities */
    for (int s=0; s<nspecies-1; ++s) 
    {
        conservative_soln[dim+2+s] = species_densities[s];
    }

    return conservative_soln;
}

// Algorithm 20b : Convert conservative to primitive
// This function has been added by Shruthi
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::convert_conservative_to_primitive ( const std::array<real,nstate> &conservative_soln ) const 
{
    /* definitions */
    std::array<real, nstate> primitive_soln;
    primitive_soln[0] = conservative_soln[0];

    const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
    for (int idim = 0; idim < dim; ++idim) {
        primitive_soln[idim+1] = vel[idim];
    }

    primitive_soln[dim+1] = compute_mixture_pressure(conservative_soln);

    const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);
    for(int ispecies = 0; ispecies < nspecies-1; ++ispecies) {
        primitive_soln[dim+2+ispecies] = mass_fractions[ispecies];
    }

    return primitive_soln;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::convert_primitive_gradient_to_conservative_gradient (
    const std::array<real,nstate> &/*primitive_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    this->pcout << "WARNING: convert_primitive_gradient_to_conservative_gradient() is not defined for current physics." << std::endl;
    this->pcout << "Aborting..." << std::endl;
    std::abort();
    return primitive_soln_gradient;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::convert_conservative_gradient_to_primitive_gradient (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    this->pcout << "WARNING: convert_conservative_gradient_to_primitive_gradient() is not defined for current physics." << std::endl;
    this->pcout << "Aborting..." << std::endl;
    std::abort();
    return conservative_soln_gradient;
}

// Algorithm 21 (f_S21): Compute species specific heat ratio
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_specific_heat_ratio ( const std::array<real,nstate> &/*conservative_soln*/ ) const
{
    std::array<real,nspecies> gamma;

    for (int s=0; s<nspecies; ++s) 
    {
        gamma[s] = (this->species_Cp[s]/this->species_Cv[s]);
    }

    return gamma;
}

template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_gamma ( const std::array<real,nstate> &conservative_soln ) const
{
    // Uses the definition given in Gouasmi thesis
    const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);

    real mixture_Cp = compute_mixture_from_species(mass_fractions,this->species_Cp);
    real mixture_Cv = compute_mixture_from_species(mass_fractions,this->species_Cv);

    real gamma = (mixture_Cp/mixture_Cv);
    return gamma;
}

// Algorithm 22 (f_S22): Compute species speed of sound
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_speed_of_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    const real temperature = compute_temperature(conservative_soln);
    const std::array<real,nspecies> gamma = compute_species_specific_heat_ratio(conservative_soln);
    std::array<real,nspecies> speed_of_sound;
    for (int s=0; s<nspecies; ++s) 
    { 
        speed_of_sound[s] = sqrt((gamma[s]*this->Rs[s]*temperature)/(this->mach_ref_sqr)); 
    }

    return speed_of_sound;
}

template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_sound ( const std::array<real,nstate> &conservative_soln ) const
{
    // This is the appropriate method for deriving mixture
    // speed of sound for thermally perfect gas as per
    // Hypersonic and High Temperature Gas Dynamics, 2nd Ed.
    // John D. Anderson
    // Chapter 14.7 Eqn 14.53
    const real R_mix = compute_mixture_gas_constant(conservative_soln);
    const real temperature = compute_temperature(conservative_soln);
    const real gamma = compute_gamma(conservative_soln);

    const real sound = sqrt(gamma*R_mix*temperature/(this->mach_ref_sqr)); 

    return sound;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Vector<double> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &duh,
    const std::vector<dealii::Tensor<2,dim> > &dduh,
    const dealii::Tensor<1,dim>               &normals,
    const dealii::Point<dim>                  &evaluation_points) const
{
    std::vector<std::string> names = post_get_names ();
    dealii::Vector<double> computed_quantities = PhysicsBase<dim,nspecies,nstate,real>::post_compute_derived_quantities_vector ( uh, duh, dduh, normals, evaluation_points);
    unsigned int current_data_index = computed_quantities.size() - 1;
    computed_quantities.grow_or_shrink(names.size());
    if constexpr (std::is_same<real,double>::value) {
        // get the solution
        std::array<double, nstate> conservative_soln;
        for (unsigned int s=0; s<nstate; ++s) {
            conservative_soln[s] = uh(s);
        }
        
        // get the solution gradient
        std::array<dealii::Tensor<1,dim,double>,nstate> conservative_soln_gradient;
        for (unsigned int s=0; s<nstate; ++s) {
            for (unsigned int d=0; d<dim; ++d) {
                conservative_soln_gradient[s][d] = duh[s][d];
            }
        }

        // Mixture density
        computed_quantities(++current_data_index) = compute_mixture_density(conservative_soln);
        // Velocities
        const dealii::Tensor<1,dim,real> vel = compute_velocities(conservative_soln);
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = vel[d];
        }
        // Mixture momentum
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = conservative_soln[1+d];
        }
        // Mixture total energy
        computed_quantities(++current_data_index) = compute_mixture_specific_total_energy(conservative_soln);
        // Mixture internal energy
        computed_quantities(++current_data_index) = compute_mixture_internal_energy(conservative_soln);
        // Mixture pressure
        computed_quantities(++current_data_index) = compute_mixture_pressure(conservative_soln);
        // Dimensional Mixture Pressure
        computed_quantities(++current_data_index) = compute_mixture_pressure(conservative_soln)*(density_ref*(u_ref*u_ref));
        // Non-dimensional temperature
        computed_quantities(++current_data_index) = compute_temperature(conservative_soln); 
        // Dimensional temperature
        computed_quantities(++current_data_index) = compute_dimensional_temperature(compute_temperature(conservative_soln));
        // Mixture specific total enthalpy
        computed_quantities(++current_data_index) = compute_mixture_specific_total_enthalpy(conservative_soln);  
        // Mass fractions
        const std::array<real,nspecies> mass_fractions = compute_mass_fractions(conservative_soln);
        for (unsigned int s=0; s<nspecies; ++s) 
        {
            computed_quantities(++current_data_index) = mass_fractions[s];
        }
        // Species densities
        const std::array<real,nspecies> species_densities = compute_species_densities(conservative_soln);
        for (unsigned int s=0; s<nspecies; ++s) 
        {
            computed_quantities(++current_data_index) = species_densities[s];
        }
    }
    if (computed_quantities.size()-1 != current_data_index) {
        this->pcout << " Did not assign a value to all the data. Missing " << computed_quantities.size() - current_data_index << " variables."
                  << " If you added a new output variable, make sure the names and DataComponentInterpretation match the above. "
                  << std::endl;
    }

    return computed_quantities;
}

template <int dim, int nspecies, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::post_get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation = PhysicsBase<dim,nspecies,nstate,real>::post_get_data_component_interpretation (); // state variables
    interpretation.push_back (DCI::component_is_scalar); // Mixture density
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Velocity
    }
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Mixture momentum
    }
    interpretation.push_back (DCI::component_is_scalar); // Mixture total energy
    interpretation.push_back (DCI::component_is_scalar); // Mixture internal energy
    interpretation.push_back (DCI::component_is_scalar); // Mixture pressure
    interpretation.push_back (DCI::component_is_scalar); // Dimensional mixture pressure
    interpretation.push_back (DCI::component_is_scalar); // Non-dimensional temperature
    interpretation.push_back (DCI::component_is_scalar); // Dimensional temperature
    interpretation.push_back (DCI::component_is_scalar); // Mixture specific total enthalpy
    for (unsigned int s=0; s<nspecies; ++s) {
         interpretation.push_back (DCI::component_is_scalar); // Mass fractions
    }
    for (unsigned int s=0; s<nspecies; ++s) {
        interpretation.push_back (DCI::component_is_scalar); // Species densities
    }

    std::vector<std::string> names = post_get_names();
    if (names.size() != interpretation.size()) {
        this->pcout << "Number of DataComponentInterpretation is not the same as number of names for output file" << std::endl;
    }
    return interpretation;
}

template <int dim, int nspecies, int nstate, typename real>
std::vector<std::string> MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::post_get_names () const
{
    std::vector<std::string> names = PhysicsBase<dim,nspecies,nstate,real>::post_get_names ();
    names.push_back ("mixture_density");
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("velocity");
    }
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("mixture_momentum");
    }
    names.push_back ("mixture_total_energy");
    names.push_back ("mixture_internal_energy");
    names.push_back ("mixture_pressure");
    names.push_back ("dimensional_mixture_pressure");
    names.push_back ("temperature");
    names.push_back ("dimensional_temperature");
    names.push_back ("mixture_specific_total_enthalpy");
    for (unsigned int s=0; s<nspecies; ++s) 
    {
      std::string string_mass_fraction = "mass_fraction";
      std::string string_species_mass_fraction = string_mass_fraction + "_" + this->species_name[s];
      names.push_back (string_species_mass_fraction);
    }
    for (unsigned int s=0; s<nspecies; ++s) 
    {
      std::string string_density = "species_density";
      std::string string_species_density = string_density + "_" + this->species_name[s];
      names.push_back (string_species_density);
    }

    return names;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::UpdateFlags MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>
::post_get_needed_update_flags () const
{
    return dealii::update_values 
            | dealii::update_gradients
            | dealii::update_quadrature_points;
}

template <int dim, int nspecies, int nstate, typename real>
MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>::MultiSpecies_ThermallyPerfect_Euler ( 
    const Parameters::AllParameters *const                    parameters_input,
    const double                                              gam_ref,
    const double                                              mach_ref,
    std::shared_ptr< ManufacturedSolutionFunction<dim,nspecies,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type_input,
    const bool                                                has_nonzero_diffusion,
    const bool                                                has_nonzero_physical_source) 
    : MultiSpecies_CaloricallyPerfect_Euler<dim,nspecies,nstate,real>(parameters_input, gam_ref, mach_ref, manufactured_solution_function, two_point_num_flux_type_input, has_nonzero_diffusion, has_nonzero_physical_source)
    , tol(1.0e-14) /// []
    , display_warning(parameters_input->display_multispecies_temperature_warnings)
{
    static_assert(nstate==dim+nspecies+1, "Physics::MultispeciesThermallyPerfect() should be created with nstate=PHILIP_DIM+PHILIP_SPECIES+1"); // Note: update this with nspecies in the future
    if(parameters_input->chemistry_input_file=="") {
        this->pcout << "Name of chemistry file containing NASA CAP data for species has not been passed in. Aborting..." << std::endl;
        std::abort(); 
    }
    this->readspeciesdata(parameters_input->chemistry_input_file);
}

// Algorithm 11 (f_M11): Compute species specific heat at constant pressure
// This function has been modified by Shruthi
// Modification: separates the temperature index into its own separate function since two different functions use it
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_specific_Cp ( const real temperature ) const
{
    if (temperature < 0.0) {
        std::cout<<"Species Cp Calculation ERROR: Temperature passed in is negative... Nondimensional Temperature = " << temperature << "...Aborting." << std::endl;
        std::abort();
    }
    real dimensional_temperature = this->compute_dimensional_temperature(temperature);
    std::array<real,nspecies> Cp;

    // species loop
    for (int ispecies=0; ispecies<nspecies; ++ispecies) 
    { 
        // main computation
        if (display_warning && (dimensional_temperature < 0.8*this->NASACAPTemperatureLimits[ispecies][0] || dimensional_temperature > 1.2*this->NASACAPTemperatureLimits[ispecies][1])) {
            std::cout << "Species Cp Calculation WARNING: Temperature exceeds the " << this->species_name[ispecies] << " polynomial limits by more than 20%..." << std::endl;
        }
        Cp[ispecies] = 0;
        for (int icoeffs = 0; icoeffs < 6; ++icoeffs) {
            Cp[ispecies] += this->Cp_poly_coeffs[ispecies][icoeffs]*pow(temperature, 5.0-icoeffs);
        }
    }

    return Cp; // nondimensional mass value
}

// Algorithm 12 (f_M12): Compute species specific heat at constant volume
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_specific_Cv ( const real temperature ) const
{
    const std::array<real,nspecies> Cp = compute_species_specific_Cp(temperature);
    std::array<real,nspecies> Cv;

    for (int s=0; s<nspecies; ++s) 
    {
        Cv[s] = Cp[s] - this->Rs[s];
    }

    return Cv; // nondimensional mass value
}

// Algorithm 13 (f_M13): Compute species specific enthalpy
// This function has been modified by Shruthi
// Modification: separates the temperature index into its own separate function since two different functions use it
// Modification #2: includes a clipping process to ensure we can still calculate for temps outside range
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_specific_enthalpy ( const real temperature ) const
{
    if (temperature < 0.0) {
        std::cout<<"Species Enthalpy Calculation ERROR: Temperature passed in is negative... Nondimensional Temperature = " << temperature << "...Aborting." << std::endl;
        std::abort();
    }
    real dimensional_temperature = this->compute_dimensional_temperature(temperature);
    std::array<real,nspecies> h;

    /// species loop
    for (int ispecies=0; ispecies<nspecies; ++ispecies) 
    { 
        // main computation
        if (display_warning && (dimensional_temperature < 0.8*this->NASACAPTemperatureLimits[ispecies][0] || dimensional_temperature > 1.2*this->NASACAPTemperatureLimits[ispecies][1])) {
            std::cout << "Species Enthalpy Calculation WARNING: Temperature exceeds the " << this->species_name[ispecies] << " polynomial limits by more than 20%..." << std::endl;
        }
        real h_ref = 0;
        for (int icoeffs = 0; icoeffs < 6; ++icoeffs)
            h_ref += this->Cp_poly_coeffs[ispecies][icoeffs]*pow(1.0, 6.0-icoeffs)*pow(6.0-icoeffs, -1.0);

        h[ispecies] = 0;
        for (int icoeffs = 0; icoeffs < 6; ++icoeffs) {
            h[ispecies] += this->Cp_poly_coeffs[ispecies][icoeffs]*pow(temperature, 6.0-icoeffs)*pow(6.0-icoeffs, -1.0);
        }
        h[ispecies] += this->species_enthalpy_offset[ispecies] - h_ref;
    }
    return h;
}

// Algorithm 14 (f_M14): Compute species specific internal energy
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_specific_internal_energy( const real temperature ) const
{
    const std::array<real,nspecies> h = compute_species_specific_enthalpy(temperature);
    std::array<real,nspecies> e;
    for (int s=0; s<nspecies; ++s) 
    {
        e[s] = (this->R_ref*this->temperature_ref/this->u_ref_sqr)*(h[s] -  this->Rs[s]*temperature);
    }

    return e;
}

// Compute the Cv integral component of species entropy (ie. \int_{T_ref}^T c_v(\tau)/\tau d\tau) using NASA polynomials
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_ThermallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_species_entropy_cv_integral ( 
    const real temperature) const
{
    if (temperature < 0.0) {
        std::cout<<"Species Entropy Calculation ERROR: Temperature passed in is negative... Nondimensional Temperature = " << temperature << "...Aborting." << std::endl;
        std::abort();
    }
    real dimensional_temperature = this->compute_dimensional_temperature(temperature);
    std::array<real,nspecies> species_entropy;
    
    /// species loop
    for (int ispecies=0; ispecies<nspecies; ++ispecies) 
    { 
        // main computation
        if (display_warning && (dimensional_temperature < 0.8*this->NASACAPTemperatureLimits[ispecies][0] || dimensional_temperature > 1.2*this->NASACAPTemperatureLimits[ispecies][1])) {
            std::cout << "Species Entropy Calculation WARNING: Temperature exceeds the " << this->species_name[ispecies] << " polynomial limits by more than 20%..." << std::endl;
        }

        species_entropy[ispecies] = 0;

        real entropy_ref = 0;
        for (int icoeffs = 0; icoeffs < 5; ++icoeffs)
            entropy_ref += this->Cp_poly_coeffs[ispecies][icoeffs]*pow(1.0, 5.0-icoeffs)*pow(5.0-icoeffs, -1.0);
        for (int icoeffs = 0; icoeffs < 5; ++icoeffs) {
            species_entropy[ispecies] += this->Cp_poly_coeffs[ispecies][icoeffs]*pow(temperature, 5.0-icoeffs)*pow(5.0-icoeffs, -1.0);
        }
        entropy_ref += this->Cp_poly_coeffs[ispecies][5]*log(1.0);
        species_entropy[ispecies] += this->Cp_poly_coeffs[ispecies][5]*log(temperature);
        species_entropy[ispecies] += this->species_entropy_offset[ispecies] - entropy_ref;
        species_entropy[ispecies] -= this->Rs[ispecies]*log(temperature);
    }

    return species_entropy;
}

// Compute species entropy by calculating integral and adding in density contribution (ie. R_k ln \rho_k)
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_ThermallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_species_entropy (
    const std::array<real,nstate> &conservative_soln) const
{
    const real temperature = compute_temperature(conservative_soln);
    const std::array<real,nspecies> species_densities = this->compute_species_densities(conservative_soln);

    std::array<real,nspecies> species_entropy = compute_species_entropy_cv_integral(temperature);
    for(int ispecies = 0; ispecies < nspecies; ispecies++) {
        species_entropy[ispecies] -= this->Rs[ispecies]*log(temperature*species_densities[ispecies]*this->density_ref);
    }

    return species_entropy;
}

// Compute Gibbs' energy of species using species entropy and species Cp
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nspecies> MultiSpecies_ThermallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_species_gibbs_energy (
    const std::array<real,nstate> &conservative_soln) const
{
    const real temperature = compute_temperature(conservative_soln);

    std::array<real,nspecies> species_entropy = compute_species_entropy(conservative_soln);
    std::array<real,nspecies> species_enthalpy = compute_species_specific_enthalpy(temperature);

    std::array<real, nspecies> species_gibbs;
    for(int ispecies = 0; ispecies < nspecies; ++ispecies) {
        species_gibbs[ispecies] = species_enthalpy[ispecies] - temperature*species_entropy[ispecies];
    }

    return species_gibbs;
}

// Compute the entropy variables from conservative solution
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> MultiSpecies_ThermallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_entropy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<real,nstate> entropy_var;
    const real temperature = compute_temperature(conservative_soln);
    std::array<real,nspecies> species_gibbs = compute_species_gibbs_energy(conservative_soln);
    real vel2 = this->compute_velocity_squared_from_conservative_solution(conservative_soln);

    entropy_var[0] = species_gibbs[nspecies-1] - (0.5*vel2);
    entropy_var[dim+1] = -1.0;

    const dealii::Tensor<1,dim,real> vel = this->compute_velocities(conservative_soln);
    for (int idim = 0; idim < dim; ++idim) {
        entropy_var[idim+1] = vel[idim];
    }

    for (int ispecies = 0; ispecies < nspecies - 1; ++ispecies) {
        entropy_var[dim+2+ispecies] = species_gibbs[ispecies] - species_gibbs[nspecies-1];
    }

    for (int istate = 0; istate < nstate; ++istate) {
        entropy_var[istate] /= temperature;
    }
    return entropy_var;
}

// Map entropy variables back to conservative solution
template <int dim, int nspecies, int nstate, typename real>
std::array<real,nstate> MultiSpecies_ThermallyPerfect_Euler<dim, nspecies, nstate, real>
::compute_conservative_variables_from_entropy_variables (
    const std::array<real,nstate> &entropy_var) const
{
    std::array<real,nstate> conservative_var;
    const real temperature = -1/entropy_var[dim+1];
    const int nth_species_idx = nspecies - 1;

    std::array<real,nspecies> species_gibbs;

    real entropy_var_vel_squared = 0.0;
    for(int idim=0; idim<dim; idim++){
        entropy_var_vel_squared += pow(entropy_var[idim + 1]*temperature, 2.0);
    }

    species_gibbs[nth_species_idx] = temperature*entropy_var[0] + entropy_var_vel_squared/2.0;
    for(int ispecies = 0; ispecies < nth_species_idx; ++ispecies) {
        species_gibbs[ispecies] = temperature*entropy_var[dim+2+ispecies] + species_gibbs[nth_species_idx];
    }

    std::array<real,nspecies> species_entropy;
    std::array<real,nspecies> species_enthalpy = compute_species_specific_enthalpy(temperature);
    for(int ispecies = 0; ispecies < nth_species_idx; ++ispecies) {
        species_entropy[ispecies] = (species_enthalpy[ispecies] - species_gibbs[ispecies])/temperature;
    }
    species_entropy[nth_species_idx] = (species_enthalpy[nth_species_idx] - species_gibbs[nth_species_idx])/temperature;

    std::array<real,nspecies> species_density;
    conservative_var[0] = 0.0;
    for(int ispecies = 0; ispecies < nspecies; ++ispecies) {
        std::array<real,nspecies> species_entropy_integral = compute_species_entropy_cv_integral(temperature);

        species_density[ispecies] = (exp((species_entropy_integral[ispecies] - species_entropy[ispecies])/(this->Rs[ispecies])))/(temperature*this->density_ref);
        conservative_var[0] += species_density[ispecies];

        if (dim + 2 + ispecies < nstate)
            conservative_var[dim+2+ispecies] = species_density[ispecies];
    }

    const real mixture_density = conservative_var[0];

    for (int idim = 0; idim < dim; ++idim) {
        conservative_var[idim+1] = mixture_density*entropy_var[idim+1]*temperature;
    }
    
    // specific kinetic energy
    const real specific_kinetic_energy = 0.50*entropy_var_vel_squared;
    // species specific enthalpy
    const std::array<real,nspecies> species_specific_enthalpy = compute_species_specific_enthalpy(temperature); 
    std::array<real,nspecies> species_specific_internal_energy;
    std::array<real,nspecies> species_specific_total_energy;
    // species energy
    for (int s=0; s<nspecies; ++s) 
    { 
      species_specific_internal_energy[s] = (this->R_ref*this->temperature_ref/this->u_ref_sqr)*(species_specific_enthalpy[s] -  this->Rs[s]*temperature);
      species_specific_total_energy[s] =  species_specific_internal_energy[s] + specific_kinetic_energy;
    }     
    // mixture energy
    real mixture_specific_total_energy = 0.0;
    for(int ispecies = 0; ispecies < nspecies; ++ispecies) {
        mixture_specific_total_energy += species_specific_total_energy[ispecies] *(species_density[ispecies]/mixture_density);
    }
    conservative_var[dim+1] = mixture_density*mixture_specific_total_energy;

    return conservative_var;
}

// Algorithm 15 (f_M15): Compute temperature
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_temperature ( const std::array<real,nstate> &conservative_soln ) const
{
    /* definitions */
    const std::array<real,nspecies> mass_fractions = this->compute_mass_fractions(conservative_soln);
    const real specific_kinetic_energy= this->compute_specific_kinetic_energy(conservative_soln);
    const real mixture_gas_constant = this->compute_mixture_gas_constant(conservative_soln);
    const real mixture_specific_total_energy = this->compute_mixture_specific_total_energy(conservative_soln);

    std::array<real,nspecies> species_specific_enthalpy;
    real mixture_specific_internal_energy;
    real mixture_specific_enthalpy;

    real f;
    std::array<real,nspecies> Cv;
    real mixture_Cv;
    real f_d; // f'
    real T_npo; // T_(n+1)
    real err = 999.9;
    int itr = 0;

    /* compute temperature using the Newton-Raphson method */
    real T_n = 2.0*this->temperature_ref; // the initial guess
    do
    {
        /// 1) f(T_n)
        // mixture specific internal energy: e = E - k
        mixture_specific_internal_energy = (mixture_specific_total_energy - specific_kinetic_energy)*this->u_ref_sqr; // dimensional value
        // species specific enthalpy at T_n
        species_specific_enthalpy = this->compute_species_specific_enthalpy(T_n/this->temperature_ref); // nondimensional mass value
        // mixture specific enthalpy at T_n
        mixture_specific_enthalpy = this->compute_mixture_from_species(mass_fractions,species_specific_enthalpy)*(this->R_ref*this->temperature_ref); // dimensional value
        // Newton-Raphson function
        f = (mixture_specific_enthalpy - mixture_gas_constant*this->R_ref* T_n) - mixture_specific_internal_energy; // dimensional value

        /// 2) f'(T_n)
        // Cv at T_n
        Cv = this->compute_species_specific_Cv(T_n/this->temperature_ref); // nondimensional mass value

        // mixture Cv
        mixture_Cv = this->compute_mixture_from_species(mass_fractions,Cv)*this->R_ref; // dimensional value

        // Newton-Raphson derivative function
        f_d = mixture_Cv;

        /// 3) main part
        T_npo = T_n - f/f_d; // dimensional value
        err = abs((T_npo-T_n)/this->temperature_ref);
        itr += 1;

        // update T
        if(itr > 9.99999e6) {
                // output temperature values for the last 10 iterations
                // included this output so user can determine if the tolerance is the issue
                std::cout << "Nearing the max iterations...iteration #" << itr << " old temperature:  " << T_n 
                            << " new temperature:  " << T_npo << std::endl;
                std::cout << " Mixture Cv:  " << mixture_Cv << std::endl << std::endl;
        }
        T_n = T_npo;
    }
    while (err>this->tol && itr < 1e7);
    if(itr == 1e7) {
        std::cout << "Maximum iterations for temperature reached without converging...Aborting..." << std::endl;
        std::abort();
    }
    T_n /= this->temperature_ref; // non-dimensional value
    if(T_n < 0) {
        std::cout << "Computed temperature is a negative value...Aborting..." << std::endl;
        std::abort();
    }
    if(T_n != T_n) {
        std::cout << "Computed temperature is NaN...Aborting..." << std::endl;
        std::abort();
    }
    return T_n;
}

// Algorithm 17 (f_M17): Compute mixture pressure
template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_mixture_pressure ( const std::array<real,nstate> &conservative_soln ) const
{
    const real mixture_density = this->compute_mixture_density(conservative_soln);
    const real mixture_gas_constant = this->compute_mixture_gas_constant(conservative_soln);
    const real temperature = compute_temperature(conservative_soln);
    const real mixture_pressure = mixture_density*mixture_gas_constant*temperature/(this->gam_ref*this->mach_ref_sqr);

    return mixture_pressure;
}

/* Supporting FUNCTIONS */
// Algorithm 20 (f_S20): Convert primitive to conservative
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nstate> MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>
::convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const 
{
    /* definitions */
    std::array<real, nstate> conservative_soln;
    const real mixture_density = this->compute_mixture_density(primitive_soln);
    std::array<real, dim> vel;

    real vel2 = 0.0;
    real sum = 0.0;
    std::array<real,nspecies> species_densities;
    std::array<real,nspecies> mass_fractions;
    const real mixture_pressure = primitive_soln[dim+1];

    /* mixture density */
    conservative_soln[0] = mixture_density;

    /* mixture momentum */
    for (int d=0; d<dim; ++d) 
    {
        vel[d] = primitive_soln[1+d];
        vel2 = vel2 + vel[d]*vel[d]; ;
        conservative_soln[1+d] = mixture_density*vel[d];
    }

    /* mixture energy */
    // mass fractions
    for (int s=0; s<nspecies-1; ++s) 
    { 
        mass_fractions[s] = primitive_soln[dim+2+s];
        sum += mass_fractions[s];
    }
    mass_fractions[nspecies-1] = 1.00 - sum;     
    // species densities
    for (int s=0; s<nspecies; ++s) 
    { 
        species_densities[s] = mixture_density*mass_fractions[s];
    }
    // mixturegas constant
    const real mixture_gas_constant = this->compute_mixture_from_species(mass_fractions,this->Rs);
    // temperature
    const real temperature = mixture_pressure/(mixture_density*mixture_gas_constant) * (this->u_ref_sqr/(this->R_ref*this->temperature_ref));
    // specific kinetic energy
    const real specific_kinetic_energy = 0.50*vel2;
    // species specific enthalpy
    const std::array<real,nspecies> species_specific_enthalpy = compute_species_specific_enthalpy(temperature); 
    // mixture enthalpy
    const real mixture_specific_enthalpy = this->compute_mixture_from_species(mass_fractions,species_specific_enthalpy);
    // mixture specific internal energy
    const real mixture_specific_internal_energy = ((this->R_ref*this->temperature_ref)/this->u_ref_sqr)*mixture_specific_enthalpy - mixture_pressure/mixture_density;
    // mixture specific total energy
    const real mixture_specific_total_energy = mixture_specific_internal_energy + specific_kinetic_energy;

    // mixture energy
    conservative_soln[dim+1] = mixture_density*mixture_specific_total_energy;

    /* species densities */
    for (int s=0; s<nspecies-1; ++s) 
    {
        conservative_soln[dim+2+s] = species_densities[s];
    }

    return conservative_soln;
}

// Algorithm 21 (f_S21): Compute species specific heat ratio
template <int dim, int nspecies, int nstate, typename real>
inline std::array<real,nspecies> MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_species_specific_heat_ratio ( const std::array<real,nstate> &conservative_soln ) const
{
    const real temperature = compute_temperature(conservative_soln);
    const std::array<real,nspecies> Cp = compute_species_specific_Cp(temperature);
    const std::array<real,nspecies> Cv = compute_species_specific_Cv(temperature);
    std::array<real,nspecies> gamma;

    for (int s=0; s<nspecies; ++s) 
    {
        gamma[s] = Cp[s]/Cv[s];
    }

    return gamma;
}

template <int dim, int nspecies, int nstate, typename real>
inline real MultiSpecies_ThermallyPerfect_Euler<dim,nspecies,nstate,real>
::compute_gamma ( const std::array<real,nstate> &conservative_soln ) const
{
    // Uses the definition given in Gouasmi thesis
    const real temperature = compute_temperature(conservative_soln);
    const std::array<real,nspecies> mass_fractions = this->compute_mass_fractions(conservative_soln);
    const std::array<real,nspecies> Cp = compute_species_specific_Cp(temperature);
    const std::array<real,nspecies> Cv = compute_species_specific_Cv(temperature);

    real mixture_Cp = this->compute_mixture_from_species(mass_fractions,Cp);
    real mixture_Cv = this->compute_mixture_from_species(mass_fractions,Cv);

    real gamma = mixture_Cp/mixture_Cv;
    return gamma;
}

///  Evaluates convective flux based on the chosen split form.
template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_ThermallyPerfect_Euler<dim, nspecies, nstate, real>
::convective_numerical_split_flux(const std::array<real,nstate> &conservative_soln1,
                                  const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    if(this->two_point_num_flux_type == two_point_num_flux_enum::KG) {
        conv_num_split_flux = this->convective_numerical_split_flux_kennedy_gruber(conservative_soln1, conservative_soln2);
    } else if(this->two_point_num_flux_type == two_point_num_flux_enum::IR) {
        std::cout << "The Ismail Roe two-point flux has not been implemented for multispecies thermally perfect gas...Aborting." << std::endl;
        std::abort();
    } else if(this->two_point_num_flux_type == two_point_num_flux_enum::CH) {
        conv_num_split_flux = convective_numerical_split_flux_chandrashekar(conservative_soln1, conservative_soln2);
    } else if(this->two_point_num_flux_type == two_point_num_flux_enum::Ra) {
        conv_num_split_flux = convective_numerical_split_flux_ranocha(conservative_soln1, conservative_soln2);
    }

    return conv_num_split_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_ThermallyPerfect_Euler<dim, nspecies, nstate, real>
::convective_numerical_split_flux_chandrashekar(const std::array<real,nstate> &conservative_soln1,
                                                 const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;

    const real temp1 = compute_temperature(conservative_soln1);
    const real temp2 = compute_temperature(conservative_soln2);

    // PULL EVERYTHING FROM HERE UNTIL ASTERISKS INTO A SEPARATE FUNCTION 
    const real temp_avg = this->compute_average(temp1, temp2);
    const real temp_sqr_avg = this->compute_average(pow(temp1,2.0),pow(temp2,2.0));
    const real temp_cubed_avg = this->compute_average(pow(temp1,3.0),pow(temp2,3.0));
    const real inv_temp_avg = 0.5*(pow(temp1,-1.0) + pow(temp2,-1.0));
    const real inv_temp_log_mean = this->compute_ismail_roe_logarithmic_mean(pow(temp1,-1.0), pow(temp2,-1.0));
    const real temp_product_operator = temp1*temp2;

    std::array<real,6> energy_flux_sum_term_coeffs = {{(2.0*temp_cubed_avg*temp_avg + temp_sqr_avg*temp_sqr_avg + 2.0*temp_sqr_avg*temp_avg*temp_avg)*(temp_product_operator/30.0),
                                                        (4.0*temp_sqr_avg*temp_avg)*(temp_product_operator/20.0),
                                                        (temp_sqr_avg + 2.0*temp_avg*temp_avg)*(temp_product_operator/12.0),
                                                        (2.0*temp_avg)*(temp_product_operator/6.0),
                                                        temp_product_operator/2.0,
                                                        1.0/inv_temp_log_mean
                                                        }};
    // ******************************************************************//
    const dealii::Tensor<1,dim,real> vel1 = this->compute_velocities(conservative_soln1);
    const dealii::Tensor<1,dim,real> vel2 = this->compute_velocities(conservative_soln2);
    dealii::Tensor<1,dim,real> vel_avg;
    dealii::Tensor<1,dim,real> vel_sqr_avg;

    for (int d=0; d<dim; ++d) {
        vel_avg[d] = 0.5*(vel1[d]+vel2[d]);
        vel_sqr_avg[d] = 0.5*(vel1[d]*vel1[d] + vel2[d]*vel2[d]);
    }

    const std::array<real,nspecies> rho_species1 = this->compute_species_densities(conservative_soln1);
    const std::array<real,nspecies> rho_species2 = this->compute_species_densities(conservative_soln2);

    // compute logarithmic mean for all components of flux except velocity component
    // and sum of mean densities for the velocity component
    // and sum of temperature monstrosity for energy component
    std::array<real, nspecies> log_mean_species_densities;
    real sum_of_log_mean_densities = 0.0;
    real pressure_diagonal = 0.0;
    std::array<real, nspecies> energy_flux_species_sum;
    // const std::array<real,nspecies> Cv = compute_species_specific_Cv(0.5*(temp1+temp2));
    for (int ispecies = 0; ispecies < nspecies; ++ispecies) {
        energy_flux_species_sum[ispecies] = 0.0;
        real h_ref = 0.0;
        // std::cout << std::endl << "species " << ispecies << " energy flux sum terms: ";
        for(int icoeff = 0; icoeff < 6; ++icoeff) {
            // Sum them terms
            energy_flux_species_sum[ispecies] += this->Cp_poly_coeffs[ispecies][icoeff]*energy_flux_sum_term_coeffs[icoeff];
            // std::cout << this->Cp_poly_coeffs[ispecies][icoeff]*energy_flux_sum_term_coeffs[icoeff] << " ";
            h_ref += this->Cp_poly_coeffs[ispecies][icoeff]*pow(1.0, 6.0-icoeff)*pow(6.0-icoeff, -1.0);
        }
        // std::cout << std::endl;
        // std::cout << "The enthalpy offset is : " << this->species_enthalpy_offset[ispecies] << " and the enthalpy at ref temp is : " << h_ref << std::endl;
        energy_flux_species_sum[ispecies] += (this->species_enthalpy_offset[ispecies] - h_ref);
        energy_flux_species_sum[ispecies] -= this->Rs[ispecies]/inv_temp_log_mean; // overleaf hasnt been updated but this is for Cp -> Cv
        // std::cout << "Total species term: " << energy_flux_species_sum[ispecies] << std::endl << std::endl;

        log_mean_species_densities[ispecies] = this->compute_ismail_roe_logarithmic_mean(rho_species1[ispecies],rho_species2[ispecies]);
        sum_of_log_mean_densities += log_mean_species_densities[ispecies];

        pressure_diagonal += this->Rs[ispecies] * (this->compute_average(rho_species1[ispecies],rho_species2[ispecies]));
    }
    // std::cout << std::endl;
    pressure_diagonal /= inv_temp_avg;
    pressure_diagonal /= (this->gam_ref*this->mach_ref_sqr);

    for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    {
        // Density equation
        conv_num_split_flux[0][flux_dim] = sum_of_log_mean_densities * vel_avg[flux_dim];

        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = sum_of_log_mean_densities*vel_avg[flux_dim]*vel_avg[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += pressure_diagonal; // Add diagonal of pressure
        
        // Species density equation
        for (int ispecies = 0; ispecies < nspecies - 1; ++ispecies) {
            const int index = dim+2+ispecies;
            conv_num_split_flux[index][flux_dim] = log_mean_species_densities[ispecies] * vel_avg[flux_dim];

            // Energy equation
            conv_num_split_flux[dim+1][flux_dim] += (energy_flux_species_sum[ispecies]*((this->R_ref*this->temperature_ref)/this->u_ref_sqr)
                                                        -0.5*vel_sqr_avg[flux_dim]) * conv_num_split_flux[index][flux_dim];
        }
        // Add last species contribution to energy flux
        conv_num_split_flux[dim+1][flux_dim] += (energy_flux_species_sum[nspecies-1]*((this->R_ref*this->temperature_ref)/this->u_ref_sqr)
                                                        -0.5*vel_sqr_avg[flux_dim])* (log_mean_species_densities[nspecies-1] * vel_avg[flux_dim]);

        // Energy equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[dim+1][flux_dim] +=  conv_num_split_flux[1+velocity_dim][flux_dim]*vel_avg[flux_dim];
        }
    }
    // std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux_kg = this->convective_numerical_split_flux_kennedy_gruber(conservative_soln1, conservative_soln2);
    // for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    // {
    //     std::cout << " kg flux ";
    //     for (int istate = 0; istate < nstate; ++istate) {
    //         std::cout << " state " << istate << " " << conv_num_split_flux_kg[istate][flux_dim];
    //     }
    //     std::cout << std::endl;
    //     std::cout << " ch flux ";
    //     for (int istate = 0; istate < nstate; ++istate) {
    //         std::cout << " state " << istate << " " << conv_num_split_flux[istate][flux_dim];
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // sleep(1);

    return conv_num_split_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> MultiSpecies_ThermallyPerfect_Euler<dim, nspecies, nstate, real>
::convective_numerical_split_flux_ranocha(const std::array<real,nstate> &conservative_soln1,
                                                 const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;

    const real temp1 = compute_temperature(conservative_soln1);
    const real temp2 = compute_temperature(conservative_soln2);

    // PULL EVERYTHING FROM HERE UNTIL ASTERISKS INTO A SEPARATE FUNCTION 
    const real temp_avg = this->compute_average(temp1, temp2);
    const real temp_sqr_avg = this->compute_average(pow(temp1,2.0),pow(temp2,2.0));
    const real temp_cubed_avg = this->compute_average(pow(temp1,3.0),pow(temp2,3.0));
    const real inv_temp_log_mean = this->compute_ismail_roe_logarithmic_mean(pow(temp1,-1.0), pow(temp2,-1.0));
    const real temp_product_operator = temp1*temp2;

    std::array<real,6> energy_flux_sum_term_coeffs = {{(2.0*temp_cubed_avg*temp_avg + temp_sqr_avg*temp_sqr_avg + 2.0*temp_sqr_avg*temp_avg*temp_avg)*(temp_product_operator/30.0),
                                                        (4.0*temp_sqr_avg*temp_avg)*(temp_product_operator/20.0),
                                                        (temp_sqr_avg + 2.0*temp_avg*temp_avg)*(temp_product_operator/12.0),
                                                        (2.0*temp_avg)*(temp_product_operator/6.0),
                                                        temp_product_operator/2.0,
                                                        1.0/inv_temp_log_mean
                                                        }};
    // ******************************************************************//
    const dealii::Tensor<1,dim,real> vel1 = this->compute_velocities(conservative_soln1);
    const dealii::Tensor<1,dim,real> vel2 = this->compute_velocities(conservative_soln2);
    dealii::Tensor<1,dim,real> vel_avg;
    dealii::Tensor<1,dim,real> vel_sqr_avg;

    for (int d=0; d<dim; ++d) {
        vel_avg[d] = 0.5*(vel1[d]+vel2[d]);
        vel_sqr_avg[d] = 0.5*(vel1[d]*vel1[d] + vel2[d]*vel2[d]);
    }

    const std::array<real,nspecies> rho_species1 = this->compute_species_densities(conservative_soln1);
    const std::array<real,nspecies> rho_species2 = this->compute_species_densities(conservative_soln2);

    // compute logarithmic mean for all components of flux except velocity component
    // and sum of mean densities for the velocity component
    // and sum of temperature monstrosity for energy component
    std::array<real, nspecies> log_mean_species_densities;
    real sum_of_log_mean_densities = 0.0;
    std::array<real, nspecies> energy_flux_species_sum;
    // const std::array<real,nspecies> Cv = compute_species_specific_Cv(0.5*(temp1+temp2));
    for (int ispecies = 0; ispecies < nspecies; ++ispecies) {
        energy_flux_species_sum[ispecies] = 0.0;
        real h_ref = 0.0;
        // std::cout << std::endl << "species " << ispecies << " energy flux sum terms: ";
        for(int icoeff = 0; icoeff < 6; ++icoeff) {
            // Sum them terms
            energy_flux_species_sum[ispecies] += this->Cp_poly_coeffs[ispecies][icoeff]*energy_flux_sum_term_coeffs[icoeff];
            // std::cout << this->Cp_poly_coeffs[ispecies][icoeff]*energy_flux_sum_term_coeffs[icoeff] << " ";
            h_ref += this->Cp_poly_coeffs[ispecies][icoeff]*pow(1.0, 6.0-icoeff)*pow(6.0-icoeff, -1.0);
        }
        // std::cout << std::endl;
        // std::cout << "The enthalpy offset is : " << this->species_enthalpy_offset[ispecies] << " and the enthalpy at ref temp is : " << h_ref << std::endl;
        energy_flux_species_sum[ispecies] += (this->species_enthalpy_offset[ispecies] - h_ref);
        energy_flux_species_sum[ispecies] -= this->Rs[ispecies]/inv_temp_log_mean; // overleaf hasnt been updated but this is for Cp -> Cv
        // std::cout << "Total species term: " << energy_flux_species_sum[ispecies] << std::endl << std::endl;

        log_mean_species_densities[ispecies] = this->compute_ismail_roe_logarithmic_mean(rho_species1[ispecies],rho_species2[ispecies]);
        sum_of_log_mean_densities += log_mean_species_densities[ispecies];

    }

    const real pressure1 = compute_mixture_pressure(conservative_soln1);
    const real pressure2 = compute_mixture_pressure(conservative_soln2);
    const real avg_pressure = this->compute_average(pressure1, pressure2);

    for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    {
        // Density equation
        conv_num_split_flux[0][flux_dim] = sum_of_log_mean_densities * vel_avg[flux_dim];

        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = sum_of_log_mean_densities*vel_avg[flux_dim]*vel_avg[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += avg_pressure; // Add diagonal of pressure
        
        // Species density equation
        for (int ispecies = 0; ispecies < nspecies - 1; ++ispecies) {
            const int index = dim+2+ispecies;
            conv_num_split_flux[index][flux_dim] = log_mean_species_densities[ispecies] * vel_avg[flux_dim];

            // Energy equation
            conv_num_split_flux[dim+1][flux_dim] += (energy_flux_species_sum[ispecies]*((this->R_ref*this->temperature_ref)/this->u_ref_sqr)
                                                        -0.5*vel_sqr_avg[flux_dim]) * conv_num_split_flux[index][flux_dim];
        }
        // Add last species contribution to energy flux
        conv_num_split_flux[dim+1][flux_dim] += (energy_flux_species_sum[nspecies-1]*((this->R_ref*this->temperature_ref)/this->u_ref_sqr)
                                                        -0.5*vel_sqr_avg[flux_dim])* (log_mean_species_densities[nspecies-1] * vel_avg[flux_dim]);

        // Energy equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[dim+1][flux_dim] +=  conv_num_split_flux[1+velocity_dim][flux_dim]*vel_avg[flux_dim];
        }

        // compute additional terms from pressure fix
        real pressure_fix = 0.25*(pressure1-pressure2)*(vel1[flux_dim]-vel2[flux_dim]);

        // Energy equation
        conv_num_split_flux[dim+1][flux_dim] -= pressure_fix;
    }
    // std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux_kg = this->convective_numerical_split_flux_kennedy_gruber(conservative_soln1, conservative_soln2);
    // for (int flux_dim = 0; flux_dim < dim; ++flux_dim)
    // {
    //     std::cout << " kg flux ";
    //     for (int istate = 0; istate < nstate; ++istate) {
    //         std::cout << " state " << istate << " " << conv_num_split_flux_kg[istate][flux_dim];
    //     }
    //     std::cout << std::endl;
    //     std::cout << " ch flux ";
    //     for (int istate = 0; istate < nstate; ++istate) {
    //         std::cout << " state " << istate << " " << conv_num_split_flux[istate][flux_dim];
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // sleep(1);

    return conv_num_split_flux;
}

// Define a sequence of possible types
#define POSSIBLE_TYPES (double)(FadType)(RadType)(FadFadType)(RadFadType)

// Define a macro to instantiate Euler and Euler functions for a specific type
#define INSTANTIATE_TYPES(r, data, type) \
    template class MultiSpecies_CaloricallyPerfect_Euler < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, type     >;\
    template class MultiSpecies_ThermallyPerfect_Euler < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, type     >;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TYPES, _, POSSIBLE_TYPES)

} // Physics namespace
} // PHiLiP namespace