/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef KTAGGINGCRITERION_HPP_
#define KTAGGINGCRITERION_HPP_

#include "Cell.hpp"
#include "DimensionDefinitions.hpp"
#include "FourthOrderDerivatives.hpp"
#include "Tensor.hpp"

class KTaggingCriterion
{
  protected:
    const double m_dx;
    const FourthOrderDerivatives m_deriv;
    // const double m_threshold_phi;
    const double m_threshold_K;

  public:
    KTaggingCriterion(double dx,  double threshold_K)
        : m_dx(dx), m_deriv(dx),
        // m_threshold_phi(threshold_phi),
        m_threshold_K(threshold_K){};

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // Tensor<1, data_t> d1_phi;
        // FOR1(idir) m_deriv.diff1(d1_phi, current_cell, idir, c_phi);

        Tensor<1, data_t> d1_K;
        FOR1(idir) m_deriv.diff1(d1_K, current_cell, idir, c_K);

        // data_t mod_d1_phi = 0;
        data_t mod_d1_K = 0;
        FOR1(idir)
        {
            // mod_d1_phi += d1_phi[idir] * d1_phi[idir];
            mod_d1_K += d1_K[idir] * d1_K[idir];
        }

        data_t criterion = m_dx * sqrt(mod_d1_K) / m_threshold_K;

        // Write back into the flattened Chombo box
        current_cell.store_vars(criterion, 0);
    }
};

#endif /* KTAGGINGCRITERION_HPP_ */
