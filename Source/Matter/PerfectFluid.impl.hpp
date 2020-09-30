/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(PERFECTFLUID_HPP_)
#error "This file should only be included through PerfectFluid.hpp"
#endif

#ifndef PERFECTFLUID_IMPL_HPP_
#define PERFECTFLUID_IMPL_HPP_
#include "DimensionDefinitions.hpp"

// Calculate the stress energy tensor elements
template <class eos_t>
template <class data_t, template <typename> class vars_t>
emtensor_t<data_t> PerfectFluid<eos_t>::compute_emtensor(
    const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
    const Tensor<2, data_t> &h_UU, const Tensor<3, data_t> &chris_ULL) const
{
    emtensor_t<data_t> out;
    Tensor<1, data_t> u_i;   // 4-velocity with lower indices

    FOR1(i)
    {
      u_i[i] = vars.Z[i] * vars.W / (vars.E + vars.D + vars.pressure);
    }


    // Calculate components of EM Tensor
    // S_ij = T_ij
    FOR2(i, j)
    {
        out.Sij[i][j] =
          vars.density * vars.enthalpy *   u_i[i] * u_i[j] +
          vars.pressure * vars.h[i][j];
    }

    // S_i (note lower index) = - n^a T_ai
    FOR1(i) { out.Si[i] =  vars.Z[i]; }

    // S = Tr_S_ij
    out.S = vars.chi * TensorAlgebra::compute_trace(out.Sij, h_UU);


    // rho = n^a n^b T_ab
    out.rho =  vars.density * vars.enthalpy * vars.W * vars.W - vars.pressure;

    return out;
}

// Adds in the RHS for the matter vars
template <class eos_t>
template <class data_t, template <typename> class vars_t,
          template <typename> class diff2_vars_t,
          template <typename> class rhs_vars_t>
void PerfectFluid<eos_t>::add_matter_rhs(
    rhs_vars_t<data_t> &total_rhs, const vars_t<data_t> &vars,
    const vars_t<Tensor<1, data_t>> &d1,
    const diff2_vars_t<Tensor<2, data_t>> &d2,
    const vars_t<data_t> &advec) const
{
    using namespace TensorAlgebra;
    const auto h_UU = compute_inverse_sym(vars.h);
    const auto chris = compute_christoffel(d1.h, h_UU);

	  total_rhs.D = 0;
    total_rhs.E = 0;

    Tensor<2, data_t> K_tensor;
    FOR2(i, j)
    {
      K_tensor[i][j] = (vars.A[i][j] + vars.h[i][j] * vars.K / 3.) / vars.chi;
    }

    total_rhs.D += advec.D + vars.lapse * vars.K * vars.D;
    total_rhs.E += advec.E + vars.lapse * vars.K *
                            (vars.pressure + vars.E);

    FOR1(i)
    {
        total_rhs.D += - vars.lapse * (d1.D[i] * vars.V[i]
                                    + vars.D * d1.V[i][i])
                       - d1.lapse[i] * vars.D * vars.V[i];

        total_rhs.E += - vars.lapse * (d1.E[i] * vars.V[i]
                                    + vars.E * d1.V[i][i])
                       - d1.lapse[i] * vars.E * vars.V[i]
                       - vars.lapse * (d1.pressure[i] * vars.V[i]
                                    + vars.pressure * d1.V[i][i])
                       - d1.lapse[i] * vars.pressure * vars.V[i]
                       - (vars.D + vars.E + vars.pressure) *
                                    vars.V[i] * d1.lapse[i];


        total_rhs.Z[i] += advec.Z[i]
                       + vars.lapse * d1.pressure[i]
                       + d1.lapse[i] * vars.pressure
                       - (vars.E + vars.D) * d1.lapse[i]
                       + vars.lapse * vars.K * vars.Z[i];
    }

    FOR2(i, j)
    {
        total_rhs.Z[i] += - vars.lapse * (d1.V[j][j] * vars.Z[i] +
                                     d1.Z[j][i] * vars.V[j])
                          - d1.lapse[j] * vars.V[j] * vars.Z[i];

        total_rhs.D += - vars.lapse * vars.D * vars.V[j] *
                            chris.ULL[i][i][j];

        total_rhs.E += - vars.lapse * vars.E * vars.V[j] *
                            chris.ULL[i][i][j]
                       - vars.lapse * vars.pressure * vars.V[j] *
                            chris.ULL[i][i][j]
                       + (vars.D + vars.E + vars.pressure) *
                            vars.lapse * vars.V[i] * vars.V[j] *
                            K_tensor[i][j];


        FOR1(k)
        {
          total_rhs.Z[i] += - vars.lapse * vars.Z[i] *
                                  vars.V[j] * chris.ULL[k][k][j]
                            + vars.lapse * vars.V[j] *
                                  vars.Z[k] * chris.ULL[k][j][i];
        }
    }
}


template <class eos_t>
template <class data_t>
void PerfectFluid<eos_t>::compute(
  Cell<data_t> current_cell) const
{

    const auto vars = current_cell.template load_vars<Vars>();
    const auto geo_vars = current_cell.template load_vars<GeoVars>();
    auto up_vars = current_cell.template load_vars<Vars>();
    //auto nw_vars = current_cell.template load_vars<Vars>();

    Tensor<1, data_t> V_i; // with lower indices: V_i

    Tensor<1, data_t> u_i; // 4-velocity with lower indices: u_i
    data_t u0 = 0.0; // 0-comp of 4-velocity (lower index)
    data_t S2 = 0.0;

    Tensor<1, data_t> residual_vec;  // residuals functions to minimize
    Tensor<1, data_t> x_vec;         // primary components to optimize
    Tensor<1, data_t> x_vec_old;
    Tensor<1, data_t> dx_vec;        // step (Newton-Rhapson)
    Tensor<2, data_t> jacobian;
    Tensor<2, data_t> cofactors;
    data_t A = vars.E + vars.D + vars.pressure;  // A = E + D + Pressure = density * enthalpy * W^2
    data_t V2 = 0.0;
    data_t V2_max = 1 - 1e-15;
    data_t kin =  vars.density * vars.energy;  // kin = density * energy
    // Inverse metric
    const auto h_UU = TensorAlgebra::compute_inverse_sym(geo_vars.h);

    data_t pressure, enthalpy, dpdrho, dpdenergy ;
     pressure = enthalpy = dpdrho = dpdenergy = 0.0;


    // Calculate V^2
    FOR2(i, j)
    {
      S2 = vars.Z[i]  * vars.Z[j] * h_UU[i][j];
    }
    while( A*A <= S2 ){  A = A * 10.; }

    A = (A > 1e100) ? 1e50 : A;
    V2 =  S2 / A / A;
    V2 = ( V2 >= 1. ) ?  V2_max : V2;
    V2 = ( V2 == V2) ?  V2: V2_max;

    x_vec[0] = A;
    x_vec[1] = V2;
    x_vec[2] = kin;



    std::cout << "START: x  " <<  x_vec[0] << " " << x_vec[1]  <<
            " " << x_vec[2]  <<'\n';





    // start Newton Rhapson manuver
    bool keep_iteration = true;
    data_t error_x = 0.0;
    data_t precision = 0.01;
    int iter, iter_extra;
    int iter_extra_max = 4;
    int iter_max = 100;

    data_t Lorentz = sqrt(1 - x_vec[1]);
    data_t det = 0.0;
    data_t dpdv2;
    iter = iter_extra = 0.0;

    // iteration starts
    while(keep_iteration){

      iter +=1;

      x_vec[0] = fabs(x_vec[0]);
      x_vec[1] = (x_vec[1] < 0.) ? 0.0 : x_vec[1];
      x_vec[1] = (x_vec[1] >= 1.) ? V2_max : x_vec[1];
      x_vec[1] = (x_vec[1] ==  x_vec[1]) ?  x_vec[1] : V2_max;
      x_vec[0] = (x_vec[0] > 1e100) ? 1e50 : x_vec[0];


      if (iter < 2) {
          // std::cout << "error_x  " <<  error_x  << "  step" <<  iter <<'\n';
          std::cout << "00: x  " <<  x_vec[0] << x_vec[1]  <<  x_vec[2]  <<'\n';
      }







      Lorentz = sqrt(1 - x_vec[1]);
      up_vars.density = vars.D / Lorentz;
      up_vars.energy = x_vec[2] / up_vars.density;
      my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);

      residual_vec[0] = S2 - x_vec[1] * x_vec[0] * x_vec[0];
      residual_vec[1] = (vars.E + vars.D) -  x_vec[1]*x_vec[0] - up_vars.density - x_vec[2];
      residual_vec[2] = x_vec[2] + pressure - x_vec[0]* (1 - x_vec[1]) + up_vars.density;

      dpdv2 = -vars.D/2/Lorentz * dpdrho + x_vec[2]/vars.D/2/Lorentz/Lorentz/Lorentz * dpdenergy;

      // Calculating Jacobian of residuals in respect of x_vec. (ie J_ij = dres[i]/dx[j])
      jacobian[0][0] = -2 * x_vec[1]*x_vec[0];
      jacobian[0][1] = - x_vec[0]*x_vec[0];
      jacobian[0][2] = 0.0;
      jacobian[1][0] = - x_vec[1];
      jacobian[1][1] = - x_vec[0] - vars.D/2/Lorentz;
      jacobian[1][2] = -1.0;
      jacobian[2][0] = - (1 - x_vec[1]);
      jacobian[2][1] =  - x_vec[0] + dpdv2 - vars.D/2/Lorentz;
      jacobian[2][2] = 1 + dpdenergy / up_vars.density;

      cofactors[0][0] =  jacobian[1][1]*jacobian[2][2] + jacobian[2][1];
      cofactors[0][1] =  jacobian[1][2]*jacobian[0][1]*jacobian[2][2];
      cofactors[0][2] =  jacobian[0][1]*jacobian[1][2];
      cofactors[1][0] =  -(jacobian[2][0] + jacobian[1][0]*jacobian[2][2]);
      cofactors[1][1] =  jacobian[0][0] * jacobian[2][2];
      cofactors[1][2] =  jacobian[0][0];
      cofactors[2][0] =  jacobian[1][0]*jacobian[2][1] - jacobian[1][1]*jacobian[2][0];
      cofactors[2][1] =  jacobian[0][1]*jacobian[2][0] - jacobian[0][0]*jacobian[2][1];
      cofactors[2][2] =  jacobian[0][0]*jacobian[1][1] - jacobian[0][1]*jacobian[1][0];

      det =  jacobian[0][0] * cofactors[0][0] + jacobian[0][1]*cofactors[1][0];

      dx_vec[0] = - (residual_vec[0]*cofactors[0][0] + residual_vec[1]*cofactors[0][1] + residual_vec[2]*cofactors[0][2])/det;
      dx_vec[1] = - (residual_vec[0]*cofactors[1][0] + residual_vec[1]*cofactors[1][1] + residual_vec[2]*cofactors[1][2])/det;
      dx_vec[2] = - (residual_vec[0]*cofactors[2][0] + residual_vec[1]*cofactors[2][1] + residual_vec[2]*cofactors[2][2])/det;



      FOR1(i){
        x_vec_old[i] = x_vec[i];
        x_vec[i] = x_vec[i] + dx_vec[i];
      }

      x_vec[0] = fabs(x_vec[0]);
      x_vec[1] = (x_vec[1] < 0.) ? 0.0 : x_vec[1];
      x_vec[1] = (x_vec[1] >= 1.) ? V2_max : x_vec[1];
      x_vec[1] = (x_vec[1] ==  x_vec[1]) ?  x_vec[1] : V2_max;
      x_vec[0] = (x_vec[0] > 1e100) ? 1e50 : x_vec[0];

      error_x = (x_vec[0] == 0.) ? fabs(dx_vec[0]) : fabs(dx_vec[0]/x_vec[0]);


      // if (iter < 4) {
      //     std::cout << "error_x  " <<  error_x  << "  step" <<  iter <<'\n';
      //     std::cout << "x  " <<  x_vec[0] << x_vec[1]  <<  x_vec[2]  <<'\n';
      // }


      if( fabs(error_x) <= precision){
        iter_extra += 1;
        keep_iteration = false;
      }
      else {
        iter_extra = 0;
      }

      if( (iter_extra >= iter_extra_max) ) {
        keep_iteration = false;
      }
      else if( iter >= iter_max) {
        keep_iteration = false;


        // std::cout << "error_x  " <<  error_x  << "  ->  Newton-Rhapson did not converge !!!" << '\n';
        //
        // std::cout << "FN error_x  " <<  error_x  << "  step" <<  iter <<'\n';
        // std::cout << "FN x  " <<  x_vec[0] << x_vec[1]  <<  x_vec[2]  <<'\n';
      }

    } // end  keep_iteration


    A = x_vec[0];
    V2 = x_vec[1];
    kin = x_vec[2];

    // Redefine variables
    up_vars.density = vars.D / Lorentz;
    up_vars.energy = kin / up_vars.density;

    my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);
    up_vars.pressure = pressure;
    up_vars.enthalpy = enthalpy;
    up_vars.W = 1.0/ sqrt(1.0 - V2);

    FOR1(i)
    {
      V_i[i] = vars.Z[i] / (vars.E + vars.D + pressure);
    }

    FOR1(i) { u_i[i] = V_i[i] * up_vars.W; }
    u0 = up_vars.W / geo_vars.lapse;

    FOR1(i) { up_vars.V[i] = u_i[i] / geo_vars.lapse / u0
                              + geo_vars.shift[i] / geo_vars.lapse;  }


    // Overwrite new values for fluid variables
    current_cell.store_vars(up_vars.density, c_density);
    current_cell.store_vars(up_vars.energy, c_energy);
    current_cell.store_vars(up_vars.pressure, c_pressure);
    current_cell.store_vars(up_vars.enthalpy, c_enthalpy);
    current_cell.store_vars(up_vars.V, GRInterval<c_V1, c_V3>());
    current_cell.store_vars(up_vars.W, c_W);
}


#endif /* PERFECTFLUID_IMPL_HPP_ */













/*

void PerfectFluid<eos_t>::compute(
  Cell<data_t> current_cell) const
{

    const auto vars = current_cell.template load_vars<Vars>();
    const auto geo_vars = current_cell.template load_vars<GeoVars>();
    auto up_vars = current_cell.template load_vars<Vars>();
    auto nw_vars = current_cell.template load_vars<Vars>();

    Tensor<1, data_t> V_i; // with lower indices: V_i
    data_t V2 = 0.0;
    Tensor<1, data_t> u_i; // 4-velocity with lower indices: u_i
    data_t u0 = 0.0; // 0-comp of 4-velocity (lower index)


    // Inverse metric
    const auto h_UU = TensorAlgebra::compute_inverse_sym(geo_vars.h);

    data_t enthalpy = 0.0;
    data_t pressure = vars.pressure;   // guess
    data_t residual = 1e6;
    data_t threshold_residual = 1e-2;                                                 // TODO: aribtrary value?
    data_t pressure_guess;
    data_t criterion;
    bool condition = true;

    int cont = 0;
    int flag = 0;
    double delta = 0.01;
    Tensor<1, data_t> V_i_nw; // with lower indices: V_i
    data_t W_nw, density_nw, energy_nw, residual_nw;
    data_t pressure_guess_nw = 0.0;

    // double fact = 1;

    // Iterative minimization of the residual to calculate the Presseure
    //  (See Alcubierre p. 245)
    while (condition) {

        pressure_guess = pressure;

        // Evaluate for guess pressure
        V2 = 10;
        while (V2 >=1) {

          FOR1(i)
          {
            V_i[i] = vars.Z[i] / (vars.E + vars.D + pressure_guess);
          }

          V2 = 0;
          FOR2(i,j)
          {
            V2 +=  V_i[i] * h_UU[i][j] * V_i[j];
          }
          pressure_guess = pressure_guess - pressure_guess * threshold_residual;
          flag += 1;
        }

        if (flag>1){
          std::cout << "it = " << cont << ",  -> V2 bigger than 1 !!!" << '\n';
          flag = 0;
        }


        up_vars.W = 1.0 / sqrt(1.0 - V2);
        up_vars.density = vars.D / up_vars.W;
        up_vars.energy = (vars.E + vars.D * ( 1 - up_vars.W)
                         + pressure_guess * (1 - up_vars.W * up_vars.W))
                         / vars.D / up_vars.W;


         // Same for guess pressure 2
        pressure_guess_nw  = pressure_guess + delta * pressure_guess;
         V2 = 10;
         while (V2 >=1) {

           FOR1(i)
           {
             V_i[i] = vars.Z[i] / (vars.E + vars.D + pressure_guess_nw);
           }

           V2 = 0;
           FOR2(i,j)
           {
             V2 +=  V_i[i] * h_UU[i][j] * V_i[j];
           }
           pressure_guess_nw = pressure_guess_nw + pressure_guess_nw * threshold_residual;

           // flag += 1;
         }

         // if (flag>1){
         //   std::cout << "!! it = " << cont << ",  -> V2 bigger than 1 !!!" << '\n';
         //   flag = 0;
         // }

         nw_vars.W = 1.0 / sqrt(1.0 - V2);
         nw_vars.density = vars.D / nw_vars.W;
         nw_vars.energy = (vars.E + vars.D * ( 1 - nw_vars.W)
                          + pressure_guess_nw * (1 - nw_vars.W * nw_vars.W))
                          / vars.D / nw_vars.W;



        // fact = 1;
        my_eos.compute_eos(pressure, enthalpy, up_vars);
        residual =  (pressure - pressure_guess);
        my_eos.compute_eos(pressure, enthalpy, nw_vars);
        residual_nw =  (pressure - pressure_guess_nw);

        if (residual > threshold_residual) {

          pressure = pressure_guess -
                     residual * (pressure_guess_nw -
                             pressure_guess) / ( residual_nw - residual );
        }

        criterion = simd_compare_gt(
                abs(residual), threshold_residual );
        condition = criterion;


        cont += 1;


        //
        // if (V2 >= 1) {
        //
        //   std::cout << "it = " << cont << ",  -> V2 bigger than 1 !!!" << '\n';
        //   //std::cout << " p = " << pressure_guess << '\n';
        //   //std::cout << " D+E = " << vars.E + vars.D << '\n';
        //   // fact *= 10;
        //   pressure = fabs(pressure_guess) * 10.;
        //   std::cout << " p (new) = " << pressure << '\n';
        //
        // }




    }

    up_vars.pressure = pressure;
    up_vars.enthalpy = enthalpy;


    FOR1(i)
    {
      V_i[i] = vars.Z[i] / (vars.E + vars.D + pressure);
    }


    FOR1(i) { u_i[i] = V_i[i] * up_vars.W; }
    u0 = up_vars.W / geo_vars.lapse;

    FOR1(i) { up_vars.V[i] = u_i[i] / geo_vars.lapse / u0
                              + geo_vars.shift[i] / geo_vars.lapse;  }


    // Overwrite new values for fluid variables
    current_cell.store_vars(up_vars.density, c_density);
    current_cell.store_vars(up_vars.energy, c_energy);
    current_cell.store_vars(up_vars.pressure, c_pressure);
    current_cell.store_vars(up_vars.enthalpy, c_enthalpy);
    current_cell.store_vars(up_vars.V, GRInterval<c_V1, c_V3>());
    current_cell.store_vars(up_vars.W, c_W);
}


*/
