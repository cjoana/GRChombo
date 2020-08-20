/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(EXT_CONSTRAINTS_HPP_)
#error "This file should only be included through ExtendedConstraints.hpp"
#endif

#ifndef EXT_CONSTRAINTS_IMPL_HPP_
#define EXT_CONSTRAINTS_IMPL_HPP_

#include "DimensionDefinitions.hpp"
#include "GRInterval.hpp"
#include "VarsTools.hpp"

inline Constraints::Constraints(double dx,
                                double cosmological_constant /*defaulted*/)
    : m_deriv(dx), m_cosmological_constant(cosmological_constant)
{
}

template <class data_t>
void Constraints::compute(Cell<data_t> current_cell) const
{
    const auto vars = current_cell.template load_vars<Vars>();
    const auto d1 = m_deriv.template diff1<Vars>(current_cell);
    const auto d2 = m_deriv.template diff2<Diff2Vars>(current_cell);

    constraints_t<data_t> out = constraint_equations(vars, d1, d2);

    // Write the rhs into the output FArrayBox
    current_cell.store_vars(out.Ham, c_Ham);
    current_cell.store_vars(out.Mom, GRInterval<c_Mom1, c_Mom3>());

    // Extended output
    // current_cell.store_vars(out.Ham_ricci, c_Ham_ricci);
    // current_cell.store_vars(out.Ham_K, c_Ham_K);
    // current_cell.store_vars(out.Ham_trA2, c_Ham_trA2);

    current_cell.store_vars(out.ricci_scalar, c_ricci_scalar);
    //current_cell.store_vars(out.ricci_scalar_tilde, c_ricci_scalar_tilde);
    current_cell.store_vars(out.rho, c_rho);
    current_cell.store_vars(out.S, c_S);
    current_cell.store_vars(out.trA2, c_trA2);
    current_cell.store_vars(out.HamRel, c_HamRel);
}

template <class data_t, template <typename> class vars_t,
          template <typename> class diff2_vars_t>
Constraints::constraints_t<data_t> Constraints::constraint_equations(
    const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
    const diff2_vars_t<Tensor<2, data_t>> &d2) const
{
    constraints_t<data_t> out;

    const data_t chi_regularised = simd_max(1e-12, vars.chi);

    auto h_UU = TensorAlgebra::compute_inverse_sym(vars.h);
    auto chris = TensorAlgebra::compute_christoffel(d1.h, h_UU);

    auto ricci = CCZ4Geometry::compute_ricci(vars, d1, d2, h_UU, chris);

    auto A_UU = TensorAlgebra::raise_all(vars.A, h_UU);
    data_t tr_A2 = TensorAlgebra::compute_trace(vars.A, A_UU);

    out.Ham = ricci.scalar +
              (GR_SPACEDIM - 1.) * vars.K * vars.K / GR_SPACEDIM - tr_A2;
    out.Ham -= 2 * m_cosmological_constant;


    Tensor<2, data_t> covd_A[CH_SPACEDIM];
    FOR3(i, j, k)
    {
        covd_A[i][j][k] = d1.A[j][k][i];
        FOR1(l)
        {
            covd_A[i][j][k] += -chris.ULL[l][i][j] * vars.A[l][k] -
                               chris.ULL[l][i][k] * vars.A[l][j];
        }
    }

    FOR1(i) { out.Mom[i] = -(GR_SPACEDIM - 1.) * d1.K[i] / GR_SPACEDIM; }
    FOR3(i, j, k)
    {
        out.Mom[i] += h_UU[j][k] *
                      (covd_A[k][j][i] - GR_SPACEDIM * vars.A[i][j] *
                                             d1.chi[k] / (2 * chi_regularised));
    }


    //Extended Output Ham
    // out.Ham_ricci = ricci.scalar;
    // out.Ham_K = (GR_SPACEDIM - 1.) * vars.K * vars.K / GR_SPACEDIM;
    // out.Ham_trA2 = - tr_A2;
    out.HamRel = pow(ricci.scalar, 2) +
              pow((GR_SPACEDIM - 1.) * vars.K * vars.K / GR_SPACEDIM, 2)
               + pow(tr_A2, 2);
    //out.HamRel += pow(2 * m_cosmological_constant, 2);
    out.ricci_scalar = ricci.scalar_tilde;
    //out.ricci_scalar_tilde = ricci.scalar_tilde;
    out.trA2 = tr_A2;

    return out;
}

#endif /* CONSTRAINTS_IMPL_HPP_ */
