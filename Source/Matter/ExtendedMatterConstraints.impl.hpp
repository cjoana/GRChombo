/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(MATTERCONSTRAINTS_HPP_)
#error "This file should only be included through MatterConstraints.hpp"
#endif

#ifndef MATTERCONSTRAINTS_IMPL_HPP_
#define MATTERCONSTRAINTS_IMPL_HPP_
#include "DimensionDefinitions.hpp"

template <class matter_t>
MatterConstraints<matter_t>::MatterConstraints(const matter_t a_matter,
                                               double dx, double G_Newton)
    : Constraints(dx, 0.0 /*No cosmological constant*/), my_matter(a_matter),
      m_G_Newton(G_Newton)
{
}

template <class matter_t>
template <class data_t>
void MatterConstraints<matter_t>::compute(Cell<data_t> current_cell) const
{
    // Load local vars and calculate derivs
    const auto vars = current_cell.template load_vars<Vars>();
    const auto d1 = m_deriv.template diff1<Vars>(current_cell);
    const auto d2 = m_deriv.template diff2<Vars>(current_cell);

    // Get the non matter terms for the constraints
    constraints_t<data_t> out = constraint_equations(vars, d1, d2);

    // Inverse metric and Christoffel symbol
    const auto h_UU = TensorAlgebra::compute_inverse_sym(vars.h);
    const auto chris = TensorAlgebra::compute_christoffel(d1.h, h_UU);

    // Energy Momentum Tensor
    const auto emtensor = my_matter.compute_emtensor(vars, d1, h_UU, chris.ULL);

    // Hamiltonain constraint
    out.Ham += -16.0 * M_PI * m_G_Newton * emtensor.rho;
    out.HamRel += pow(16.0 * M_PI * m_G_Newton * emtensor.rho, 2);
    out.HamRel = (out.Ham * out.Ham) / out.HamRel ;
    out.rho = emtensor.rho;
    out.S = emtensor.S;

    // Momentum constraints
    FOR1(i) { out.Mom[i] += -8.0 * M_PI * m_G_Newton * emtensor.Si[i]; }

    // Write the rhs into the output FArrayBox
    current_cell.store_vars(out.Ham, c_Ham);
    current_cell.store_vars(out.Mom, GRInterval<c_Mom1, c_Mom3>());

    // Extended output
    // current_cell.store_vars(out.Ham_ricci, c_Ham_ricci);
    // current_cell.store_vars(out.Ham_K, c_Ham_K);
    // current_cell.store_vars(out.Ham_trA2, c_Ham_trA2);
    // current_cell.store_vars(out.Ham_rho, c_Ham_rho); 
    current_cell.store_vars(out.ricci_scalar, c_ricci_scalar);
    //current_cell.store_vars(out.ricci_scalar_tilde, c_ricci_scalar_tilde);
    current_cell.store_vars(out.rho, c_rho);
    current_cell.store_vars(out.trA2, c_trA2);
    current_cell.store_vars(out.S, c_S);
    current_cell.store_vars(out.HamRel, c_HamRel);
}

#endif /* MATTERCONSTRAINTS_IMPL_HPP_ */
