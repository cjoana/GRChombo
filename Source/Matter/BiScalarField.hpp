/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef BISCALARFIELD_HPP_
#define BISCALARFIELD_HPP_

#include "CCZ4Geometry.hpp"
#include "DefaultPotential.hpp"
#include "FourthOrderDerivatives.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS, total num of components
#include "VarsTools.hpp"

//!  Calculates the matter type specific elements such as the EMTensor and
//   matter evolution
/*!
     This class is an example of a matter_t object which calculates the
     matter type specific elements for the RHS update and the evaluation
     of the constraints. This includes the Energy Momentum Tensor, and
     the matter evolution terms. In this case, a scalar field,
     the matter elements are phi and (minus) its conjugate momentum, Pi.
     It is templated over a potential function potential_t which the
     user must specify in a class, although a default is provided which
     sets dVdphi and V_of_phi to zero.
     It assumes minimal coupling of the field to gravity.
     \sa MatterCCZ4(), ConstraintsMatter()
*/
template <class potential_t = DefaultPotential> class ScalarField
{
  protected:
    //! The local copy of the potential
    potential_t my_potential;

  public:
    //!  Constructor of class ScalarField, inputs are the matter parameters.
    ScalarField(const potential_t a_potential) : my_potential(a_potential) {}

    //! Structure containing the variables for the matter fields
    template <class data_t> struct SFObject
    {
        data_t phi;
        data_t Pi;

        data_t phi2;
        data_t Pi2;

    };

    //! Structure containing the rhs variables for the matter fields
    template <class data_t> struct Vars
    {
        data_t phi;
        data_t Pi;

        data_t phi2;
        data_t Pi2;

        /// Defines the mapping between members of Vars and Chombo grid
        /// variables (enum in User_Variables)
        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            VarsTools::define_enum_mapping(mapping_function, c_phi, phi);
            VarsTools::define_enum_mapping(mapping_function, c_Pi, Pi);

            VarsTools::define_enum_mapping(mapping_function, c_phi2, phi2);
            VarsTools::define_enum_mapping(mapping_function, c_Pi2, Pi2);
        }
    };

    //! Structure containing the rhs variables for the matter fields requiring
    //!  2nd derivs
    template <class data_t> struct Diff2Vars
    {
        data_t phi;

        data_t phi2;

        /// Defines the mapping between members of Vars and Chombo grid
        ///  variables (enum in User_Variables)
        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            VarsTools::define_enum_mapping(mapping_function, c_phi, phi);
            VarsTools::define_enum_mapping(mapping_function, c_phi2, phi2);
        }
    };

    //! The function which calculates the EM Tensor, given the vars and
    //! derivatives, including the potential
    template <class data_t, template <typename> class vars_t>
    emtensor_t<data_t> compute_emtensor(
        const vars_t<data_t> &vars,          //!< the value of the variables
        const vars_t<Tensor<1, data_t>> &d1, //!< the value of the 1st derivs
        const Tensor<2, data_t> &h_UU, //!< the inverse metric (raised indices)
        const Tensor<3, data_t> &chris_ULL)
        const; //!< the conformal christoffel symbol

    //! The function which calculates the EM Tensor, given the vars and
    //! derivatives, excluding the potential
    template <class data_t, template <typename> class vars_t>
    static void emtensor_excl_potential(
        emtensor_t<data_t> &out,         //!< the em tensor output
        const vars_t<data_t> &vars,      //!< the value of the variables
        const SFObject<data_t> &vars_sf, //!< the value of the sf variables
        const Tensor<1, data_t>
            &d1_phi,                   //!< the value of the first deriv of phi
        const Tensor<1, data_t>
            &d1_phi2,                  //!< the value of the first deriv of phi
        const Tensor<2, data_t> &h_UU, //!< the inverse metric (raised indices).
        const Tensor<3, data_t>
            &chris_ULL); //!< the conformal christoffel symbol

    //! The function which adds in the RHS for the matter field vars,
    //! including the potential
    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t,
              template <typename> class rhs_vars_t>
    void add_matter_rhs(
        rhs_vars_t<data_t> &total_rhs,       //!< value of the RHS for all vars
        const vars_t<data_t> &vars,          //!< value of the variables
        const vars_t<Tensor<1, data_t>> &d1, //!< value of the 1st derivs
        const diff2_vars_t<Tensor<2, data_t>> &d2, //!< value of the 2nd derivs
        const vars_t<data_t> &advec)
        const; //!< the value of the advection terms

    //! The function which calculates the RHS for the matter field vars
    //! excluding the potential
    template <class data_t, template <typename> class vars_t>
    static void matter_rhs_excl_potential(
        SFObject<data_t>
            &rhs_sf, //!< the value of the RHS terms for the sf vars
        const vars_t<data_t> &vars,      //!< the values of all the variables
        const SFObject<data_t> &vars_sf, //!< the value of the sf variables
        const vars_t<Tensor<1, data_t>> &d1, //!< the value of the 1st derivs
        const Tensor<1, data_t> &d1_phi, //!< the value of the 1st derivs of phi
        const Tensor<2, data_t> &d2_phi, //!< the value of the 2nd derivs of phi
        const Tensor<1, data_t> &d1_phi2, //!< the value of the 1st derivs of phi
        const Tensor<2, data_t> &d2_phi2, //!< the value of the 2nd derivs of phi
        const SFObject<data_t> &advec_sf); //!< advection terms for the sf vars
};

#include "BiScalarField.impl.hpp"

#endif /* BISCALARFIELD_HPP_ */