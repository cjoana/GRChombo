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
          vars.pressure * vars.h[i][j]/vars.chi;
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

    Tensor<1, data_t> x_vec;         // primary components to optimize

    // Tensor<2, data_t> cofactors;
    data_t A = vars.E + vars.D + vars.pressure;  // A = E + D + Pressure = density * enthalpy * W^2
    data_t V2 = 0.0;
    data_t V2_max = 1 - 1e-15;
    data_t kin = 0;  // =  vars.density * vars.energy;  // kin = density * energy
    data_t omega = 0;
    // Inverse metric
    const auto h_UU = TensorAlgebra::compute_inverse_sym(geo_vars.h);

    data_t pressure, enthalpy, dpdrho, dpdenergy ;
    pressure = enthalpy = dpdrho = dpdenergy = 0.0;


    // Calculate V^2
    S2 = 0.0;
    FOR2(i, j)
    {
      S2 += vars.Z[i] * vars.Z[j] * h_UU[i][j]; // * geo_vars.chi;
    }


    if (S2<0){
      std::cout << "    wrong stared S2 ::  " <<  S2  << '\n';
      std::cout << "    lapse  " << geo_vars.lapse   <<'\n';

      std::cout << "    metric  " <<'\n';
      std::cout << "    1i " <<  geo_vars.h[0][0] << " " << geo_vars.h[0][1]  << " " << geo_vars.h[0][2]<<'\n';
      std::cout << "    2i " <<  geo_vars.h[1][0] << " " << geo_vars.h[1][1]  << " " << geo_vars.h[1][2]<<'\n';
      std::cout << "    3i " <<  geo_vars.h[2][0] << " " << geo_vars.h[2][1]  << " " << geo_vars.h[2][2]<<'\n';


    }



    S2 = (S2 < 0) ? 0.0 : S2;
    // S2 = fabs(S2);
    while( A*A <= S2 ){  A = A * 10.; }
    A = (A > 1e20) ? 1e20 : A;
    V2 =  S2 / A / A;
    // V2 = - (vars.D - vars.density) * (vars.D + vars.density) / vars.density/ vars.density;
    V2 = ( V2 >= 1. ) ?  V2_max : V2;
    V2 = ( V2 == V2) ?  V2: V2_max;
    V2 = ( S2 == 0. ) ?  0.0 : V2;

    // up_vars.density = vars.D * sqrt(1 -  V2);
    // kin = up_vars.density * vars.energy;
    // kin = vars.density * vars.energy;

    x_vec[0] = A;
    x_vec[1] = V2;
    x_vec[2] = 0;





    // DEBUG
    // -----------------
    if (!( (vars.density  == vars.density) || (vars.energy == vars.density)
            || (kin == kin)) || (kin > 1e200) ) {

      std::cout << "BAD START: x  " <<  vars.density << " " << vars.energy  <<
              " " << kin  <<'\n';

      std::cout << "    D, E, W ::  " <<  vars.D << " " << vars.E  <<
                      " " << vars.W  <<'\n';
    }





    // std::cout << "x_vec ini  " <<  x_vec[0]  << " " << x_vec[1] << '\n';  //DEBUG delete

    recover_primvars_NR2D(current_cell, x_vec, S2);

    // Delete
    Tensor<1, data_t> x_vecf1;
    FOR1(i) { x_vecf1[i] = x_vec[i];}

    // Delete
    x_vec[0] = A;
    x_vec[1] = V2;
    x_vec[2] = 0;



//###############################################################################

    // start Newton Rhapson manuver
    bool keep_iteration = true;
    data_t error_x = 0.0;
    data_t precision = 1e-10;
    data_t precision_2 = 1e-16;
    int iter, iter_extra;
    int iter_extra_max = 4;
    int iter_max = 1e6;

    Tensor<1, data_t> residual_vec;  // residuals functions to minimize
    Tensor<1, data_t> x_vec_old;
    Tensor<1, data_t> x_vec_orig;    //old
    Tensor<1, data_t> dx_vec;        // step (Newton-Rhapson)
    Tensor<2, data_t> jacobian;

    data_t Lorentz = sqrt(1 - x_vec[1]);
    data_t det = 0.0;
    data_t dpdv2;
    iter = iter_extra = 0.0;

    up_vars.pressure = vars.pressure;
    up_vars.energy = vars.energy;

    x_vec_orig[0] = A;
    x_vec_orig[1] = V2;
    x_vec_orig[2] = 0;

// print
    // std::cout << "00 : resids " <<  residual_vec[0] << residual_vec[1]  <<  residual_vec[2]  <<'\n';

    // iteration starts
    while(keep_iteration){

      iter +=1;

      x_vec[0] = fabs(x_vec[0]);
      x_vec[1] = (x_vec[1] < 0.) ? 0.0 : x_vec[1];
      x_vec[1] = (x_vec[1] >= 1.) ? V2_max : x_vec[1];
      x_vec[1] = (x_vec[1] ==  x_vec[1]) ?  x_vec[1] : x_vec_old[1];
      x_vec[0] = (x_vec[0] > 1e20) ? x_vec_old[0] : x_vec[0];


      // if (iter < 2) {
      //     // std::cout << "error_x  " <<  error_x  << "  step" <<  iter <<'\n';
      //     std::cout << "00: x  " <<  x_vec[0] << x_vec[1]  <<  x_vec[2]  <<'\n';
      // }


      Lorentz = sqrt(1 - x_vec[1]);
      up_vars.W = 1.0/Lorentz;
      up_vars.density = vars.D / up_vars.W;

      up_vars.energy = 0;
      my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);
      omega = dpdenergy/up_vars.density;
      pressure = omega / (omega+1) * (A*(1 - V2)  - up_vars.density);
      up_vars.energy = (A*(1 - V2) - (up_vars.density + pressure))/ up_vars.density;
      // up_vars.energy = (vars.E + vars.D * ( 1 - up_vars.W)
      //                  + pressure * (1 - up_vars.W * up_vars.W))
      //                  / vars.D / up_vars.W;
      // my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);


      residual_vec[0] = S2 - x_vec[1] * x_vec[0] * x_vec[0];
      residual_vec[1] =  (vars.E + vars.D) -  x_vec[0] + pressure;


// print
      // if( iter > 10){
        // std::cout << iter << " : resids " << "  " <<  residual_vec[0] << "  " << residual_vec[1]  << "  " <<  residual_vec[2]  <<'\n';
      // }

      // Calculating Jacobian of residuals in respect of x_vec. (ie J_ij = dres[i]/dx[j])
      jacobian[0][0] = -2*x_vec[1]*x_vec[0];
      jacobian[0][1] = - x_vec[0]*x_vec[0];
      jacobian[1][0] = -1 +  omega / (omega+1) * (1 - x_vec[1]);
      jacobian[1][1] = omega / (omega+1) *( vars.D /2/Lorentz  - x_vec[0]);

      det =  jacobian[0][0]*jacobian[1][1] - jacobian[1][0] * jacobian[0][1];

      dx_vec[0] = - (residual_vec[0]*jacobian[1][1] - residual_vec[1]*jacobian[0][1])/det;
      dx_vec[1] =  (residual_vec[0]*jacobian[1][0] - residual_vec[1]*jacobian[0][0])/det;



      FOR1(i){
        x_vec_old[i] = x_vec[i];
        x_vec[i] = x_vec[i] + dx_vec[i];
      }

      x_vec[0] = fabs(x_vec[0]);
      x_vec[0] = (x_vec[0] ==  x_vec[0]) ?  x_vec[0] : x_vec_old[0];
      x_vec[0] = (x_vec[0] > 1e20) ? x_vec_old[0] : x_vec[0];  // Assuming planckian units!!
      x_vec[1] = (x_vec[1] < 0.) ? 0.0 : x_vec[1];
      x_vec[1] = (x_vec[1] >= 1.) ? V2_max : x_vec[1];
      x_vec[1] = (x_vec[1] ==  x_vec[1]) ?  x_vec[1] : x_vec_old[1];

      error_x = (x_vec[0] == 0.) ? fabs(dx_vec[0]) : fabs(dx_vec[0]/x_vec[0]);


      // if (iter < 4) {
      //     std::cout << "error_x  " <<  error_x  << "  step" <<  iter <<'\n';
      //     std::cout << "x  " <<  x_vec[0] << x_vec[1]  <<  x_vec[2]  <<'\n';
      // }


      if( (fabs(error_x) <= precision)  || residual_vec[0] <= precision_2){
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

        // DEBUG
        // -------------------------
        std::cout << "error_x  " <<  error_x  << "  ->  Newton-Rhapson did not converge !!!" << '\n';
        std::cout << "FN dx[0]  " <<  dx_vec[0]  << "  step  "    <<  1.*iter <<'\n';
        std::cout << "FN x  " <<  "  " << x_vec[0] <<  "  " << x_vec[1]  << "  " <<  x_vec[2]  <<'\n';
        std::cout << "Orig x  " <<  "  " << x_vec_orig[0] <<  "  " << x_vec_orig[1]  << "  " <<  x_vec_orig[2]  <<'\n';

        std::cout << " S2  " <<  "  " << S2 << "   chi  " << geo_vars.chi <<  "\n";

        std::cout << " diff pressure  " <<  x_vec[0] - vars.D - vars.E <<   "   press: " <<  pressure
                  << " diff " <<  x_vec[0] - vars.D - vars.E  - pressure <<   "\n" << "\n";

      }

    } // end  keep_iteration


//###############################################################################



    // Delete
    Tensor<1, data_t> x_vecf2;
    FOR1(i) { x_vecf2[i] = x_vec[i];}
    if ( !( (x_vecf1[0] == x_vecf2[0]) || (x_vecf1[1] == x_vecf2[1])) ){
        std::cout << "x_vec is different!! : " << "\n";
        std::cout <<  A << " " <<  V2  << "\n";
        std::cout <<  x_vecf1[0] << " " << x_vecf1[1]  << "\n";
        std::cout <<  x_vecf2[0] << " " << x_vecf2[1]  << "\n";
    }


    A = x_vec[0];
    V2 = x_vec[1];


    //data_t Lorentz;
    // Redefine variables
    Lorentz = sqrt(1 - V2);
    up_vars.density = vars.D / Lorentz;
    pressure = omega / (omega+1) * (A*(1 - V2)  - up_vars.density);
    up_vars.energy = (A*(1 - V2) - (up_vars.density + pressure))/ up_vars.density;
    my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);
    up_vars.pressure = pressure;
    up_vars.enthalpy = enthalpy;

    /*
    up_vars.energy = (vars.E + vars.D * ( 1 - up_vars.W)
                     + pressure * (1 - up_vars.W * up_vars.W))
                     / vars.D / up_vars.W;

    my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);
    up_vars.pressure = pressure;
    up_vars.enthalpy = enthalpy;
    up_vars.W = 1.0/ sqrt(1.0 - V2);


    // FOR1(i)
    {
      V_i[i] = vars.Z[i] / (vars.E + vars.D + pressure);
    }

    FOR1(i) { u_i[i] = V_i[i] * up_vars.W; }
    u0 = up_vars.W / geo_vars.lapse;

    FOR1(i) { up_vars.V[i] = u_i[i] / geo_vars.lapse / u0
                              + geo_vars.shift[i] / geo_vars.lapse;  }
    */

    FOR1(i) {
       up_vars.V[i] = 0;
     }
    FOR2(i,j) {
      up_vars.V[i] += vars.Z[j] * h_UU[i][j] * geo_vars.chi / (vars.E + vars.D + pressure);
    }


    // Overwrite new values for fluid variables
    current_cell.store_vars(up_vars.density, c_density);
    current_cell.store_vars(up_vars.energy, c_energy);
    current_cell.store_vars(up_vars.pressure, c_pressure);
    current_cell.store_vars(up_vars.enthalpy, c_enthalpy);
    current_cell.store_vars(up_vars.V, GRInterval<c_V1, c_V3>());
    current_cell.store_vars(up_vars.W, c_W);
}






template <class eos_t>
template <class data_t>
void PerfectFluid<eos_t>::recover_primvars_NR2D(Cell<data_t> current_cell,
                                                Tensor<1, data_t> &x_vec,
                                                const data_t &S2) const
{




    const auto vars = current_cell.template load_vars<Vars>();
    auto up_vars = current_cell.template load_vars<Vars>();

    Tensor<1, data_t> residual_vec;  // residuals functions to minimize
    Tensor<1, data_t> x_vec_old;
    Tensor<1, data_t> x_vec_orig;    //old
    Tensor<1, data_t> dx_vec;        // step (Newton-Rhapson)
    Tensor<2, data_t> jacobian;

    data_t omega = 0;

    data_t A = x_vec[0] ;
    data_t V2 = x_vec[1] ;
    data_t kin = x_vec[2];


    data_t pressure, enthalpy, dpdrho, dpdenergy ;
    pressure = enthalpy = dpdrho = dpdenergy = 0.0;

    // start Newton Rhapson manuver
    bool keep_iteration = true;
    data_t error_x = 0.0;
    data_t precision = 1e-10;
    data_t precision_2 = 1e-16;
    data_t V2_max = 1 - 1e-15;
    int iter, iter_extra;
    int iter_extra_max = 4;
    int iter_max = 1e6;

    data_t Lorentz = sqrt(1 - x_vec[1]);
    data_t det = 0.0;
    iter = iter_extra = 0.0;
    data_t dpdv2;

    up_vars.pressure = vars.pressure;
    up_vars.energy = vars.energy;

    x_vec_orig[0] = x_vec[0];
    x_vec_orig[1] = x_vec[1];
    x_vec_orig[2] = x_vec[2];
    x_vec_old[0] = x_vec[0];
    x_vec_old[1] = x_vec[1];
    x_vec_old[2] = x_vec[2];

    // print check
    // std::cout << "00 : resids " <<  residual_vec[0] << residual_vec[1]  <<  residual_vec[2]  <<'\n';

    // iteration starts
    while(keep_iteration){

        iter +=1;

        x_vec[0] = fabs(x_vec[0]);
        x_vec[0] = (x_vec[0] > 1e20) ? x_vec_old[0] : x_vec[0];
        x_vec[1] = (x_vec[1] < 0.) ? 0.0 : x_vec[1];
        x_vec[1] = (x_vec[1] >= 1.) ? V2_max : x_vec[1];
        x_vec[0] = (x_vec[0] ==  x_vec[0]) ?  x_vec[0] : x_vec_old[0];
        x_vec[1] = (x_vec[1] ==  x_vec[1]) ?  x_vec[1] : x_vec_old[1];


        Lorentz = sqrt(1 - x_vec[1]);
        up_vars.W = 1.0/Lorentz;
        up_vars.density = vars.D / up_vars.W;

        up_vars.energy = 0;
        my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);
        omega = dpdenergy/up_vars.density;
        pressure = omega / (omega+1) * (A*(1 - V2)  - up_vars.density);
        up_vars.energy = (A*(1 - V2) - (up_vars.density + pressure))/ up_vars.density;
        // up_vars.energy = (vars.E + vars.D * ( 1 - up_vars.W)
        //                  + pressure * (1 - up_vars.W * up_vars.W))
        //                  / vars.D / up_vars.W;
        // my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);


        residual_vec[0] = S2 - x_vec[1] * x_vec[0] * x_vec[0];
        residual_vec[1] =  (vars.E + vars.D) -  x_vec[0] + pressure;


        // print check
        // if( iter > 10){
        // std::cout << iter << " : resids " << "  " <<  residual_vec[0] << "  " << residual_vec[1]  << "  " <<  residual_vec[2]  <<'\n';
        // }

        // Calculating Jacobian of residuals in respect of x_vec. (ie J_ij = dres[i]/dx[j])
        jacobian[0][0] = -2*x_vec[1]*x_vec[0];
        jacobian[0][1] = - x_vec[0]*x_vec[0];
        jacobian[1][0] = -1 +  omega / (omega+1) * (1 - x_vec[1]);
        jacobian[1][1] = omega / (omega+1) *( vars.D /2/Lorentz  - x_vec[0]);

        det =  jacobian[0][0]*jacobian[1][1] - jacobian[1][0] * jacobian[0][1];

        dx_vec[0] = - (residual_vec[0]*jacobian[1][1] - residual_vec[1]*jacobian[0][1])/det;
        dx_vec[1] =  (residual_vec[0]*jacobian[1][0] - residual_vec[1]*jacobian[0][0])/det;


        // Advance in x_vec
        FOR1(i){
            x_vec_old[i] = x_vec[i];
            x_vec[i] = x_vec[i] + dx_vec[i];
        }


        x_vec[0] = fabs(x_vec[0]);
        x_vec[0] = (x_vec[0] ==  x_vec[0]) ?  x_vec[0] : x_vec_old[0];
        x_vec[0] = (x_vec[0] > 1e20) ? x_vec_old[0] : x_vec[0];
        x_vec[1] = (x_vec[1] < 0.) ? 0.0 : x_vec[1];
        x_vec[1] = (x_vec[1] >= 1.) ? V2_max : x_vec[1];
        x_vec[1] = (x_vec[1] ==  x_vec[1]) ?  x_vec[1] : x_vec_old[1];

        error_x = (x_vec[0] == 0.) ? fabs(dx_vec[0]) : fabs(dx_vec[0]/x_vec[0]);


        // if (iter < 4) {
        //     std::cout << "error_x  " <<  error_x  << "  step" <<  iter <<'\n';
        //     std::cout << "x  " <<  x_vec[0] << x_vec[1]  <<  x_vec[2]  <<'\n';
        // }


        if( (fabs(error_x) <= precision)  || residual_vec[0] <= precision_2){
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

            // const auto vars = current_cell.template load_vars<Vars>();
            const auto geo_vars = current_cell.template load_vars<GeoVars>();

            // DEBUG
            // -------------------------
            std::cout << "error_x  " <<  error_x  << "  ->  Newton-Rhapson did not converge !!!" << '\n';
            std::cout << "FN dx[0]  " <<  dx_vec[0]  << "  step  "    <<  1.*iter <<'\n';
            std::cout << "FN x  " <<  "  " << x_vec[0] <<  "  " << x_vec[1]  << "  " <<  x_vec[2]  <<'\n';
            std::cout << "Orig x  " <<  "  " << x_vec_orig[0] <<  "  " << x_vec_orig[1]  << "  " <<  x_vec_orig[2]  <<'\n';

            std::cout << " S2  " <<  "  " << S2 << "   chi  " << geo_vars.chi <<  "\n";

            std::cout << " diff pressure  " <<  x_vec[0] - vars.D - vars.E <<   "   press: " <<  pressure
                      << " diff " <<  x_vec[0] - vars.D - vars.E  - pressure <<   "\n" << "\n";

        }

    } // end  keep_iteration

}



template <class eos_t>
template <class data_t>
void PerfectFluid<eos_t>::recover_primvars_NR3D(Cell<data_t> current_cell,
                                                Tensor<1, data_t> &x_vec,
                                                const data_t &S2) const
{
//    const auto vars = current_cell.template load_vars<Vars>();
//    auto up_vars = current_cell.template load_vars<Vars>();
//
//    Tensor<1, data_t> residual_vec;  // residuals functions to minimize
//    Tensor<1, data_t> x_vec_old;
//    Tensor<1, data_t> x_vec_orig;    //old
//    Tensor<1, data_t> dx_vec;        // step (Newton-Rhapson)
//    Tensor<2, data_t> jacobian;
//
//    data_t omega = 0;
//
//    data_t A = x_vec[0] ;
//    data_t V2 = x_vec[1] ;
//    data_t kin = x_vec[2];
//
//
//    data_t pressure, enthalpy, dpdrho, dpdenergy ;
//    pressure = enthalpy = dpdrho = dpdenergy = 0.0;
//
//    // start Newton Rhapson manuver
//    bool keep_iteration = true;
//    data_t error_x = 0.0;
//    data_t precision = 1e-10;
//    data_t precision_2 = 1e-16;
//    data_t V2_max = 1 - 1e-15;
//    int iter, iter_extra;
//    int iter_extra_max = 4;
//    int iter_max = 1e6;
//
//    data_t Lorentz = sqrt(1 - x_vec[1]);
//    data_t det = 0.0;
//    iter = iter_extra = 0.0;
//    data_t dpdv2;
//
//    up_vars.pressure = vars.pressure;
//    up_vars.energy = vars.energy;
//
//    x_vec_orig[0] = x_vec[0];
//    x_vec_orig[1] = x_vec[1];
//    x_vec_orig[2] = x_vec[2];
//
//    // print check
//    // std::cout << "00 : resids " <<  residual_vec[0] << residual_vec[1]  <<  residual_vec[2]  <<'\n';
//
//    // iteration starts
//    while(keep_iteration){
//
//        iter +=1;
//
//        x_vec[0] = fabs(x_vec[0]);
//        x_vec[1] = (x_vec[1] < 0.) ? 0.0 : x_vec[1];
//        x_vec[1] = (x_vec[1] >= 1.) ? V2_max : x_vec[1];
//        x_vec[1] = (x_vec[1] ==  x_vec[1]) ?  x_vec[1] : x_vec_old[1];
//        x_vec[0] = (x_vec[0] > 1e20) ? x_vec_old[0] : x_vec[0];
//
//
//        // if (iter < 2) {
//        //     // std::cout << "error_x  " <<  error_x  << "  step" <<  iter <<'\n';
//        //     std::cout << "00: x  " <<  x_vec[0] << x_vec[1]  <<  x_vec[2]  <<'\n';
//        // }
//
//
//        Lorentz = sqrt(1 - x_vec[1]);
//        up_vars.W = 1.0/Lorentz;
//        up_vars.density = vars.D / up_vars.W;
//
//        up_vars.energy = 0;
//        my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);
//        omega = dpdenergy/up_vars.density;
//        pressure = omega / (omega+1) * (A*(1 - V2)  - up_vars.density);
//        up_vars.energy = (A*(1 - V2) - (up_vars.density + pressure))/ up_vars.density;
//        // up_vars.energy = (vars.E + vars.D * ( 1 - up_vars.W)
//        //                  + pressure * (1 - up_vars.W * up_vars.W))
//        //                  / vars.D / up_vars.W;
//        // my_eos.compute_eos(pressure, enthalpy, dpdrho, dpdenergy, up_vars);
//
//
//        residual_vec[0] = S2 - x_vec[1] * x_vec[0] * x_vec[0];
//        residual_vec[1] =  (vars.E + vars.D) -  x_vec[0] + pressure;
//
//
//        // print
//        // if( iter > 10){
//        // std::cout << iter << " : resids " << "  " <<  residual_vec[0] << "  " << residual_vec[1]  << "  " <<  residual_vec[2]  <<'\n';
//        // }
//
//        // Calculating Jacobian of residuals in respect of x_vec. (ie J_ij = dres[i]/dx[j])
//        jacobian[0][0] = -2*x_vec[1]*x_vec[0];
//        jacobian[0][1] = - x_vec[0]*x_vec[0];
//        jacobian[1][0] = -1 +  omega / (omega+1) * (1 - x_vec[1]);
//        jacobian[1][1] = omega / (omega+1) *( vars.D /2/Lorentz  - x_vec[0]);
//
//        det =  jacobian[0][0]*jacobian[1][1] - jacobian[1][0] * jacobian[0][1];
//
//        dx_vec[0] = - (residual_vec[0]*jacobian[1][1] - residual_vec[1]*jacobian[0][1])/det;
//        dx_vec[1] =  (residual_vec[0]*jacobian[1][0] - residual_vec[1]*jacobian[0][0])/det;
//
//
//
//        FOR1(i){
//            x_vec_old[i] = x_vec[i];
//            x_vec[i] = x_vec[i] + dx_vec[i];
//        }
//
//        x_vec[0] = fabs(x_vec[0]);
//        x_vec[0] = (x_vec[0] ==  x_vec[0]) ?  x_vec[0] : x_vec_old[0];
//        x_vec[0] = (x_vec[0] > 1e20) ? x_vec_old[0] : x_vec[0];  // Assuming planckian units!!
//        x_vec[1] = (x_vec[1] < 0.) ? 0.0 : x_vec[1];
//        x_vec[1] = (x_vec[1] >= 1.) ? V2_max : x_vec[1];
//        x_vec[1] = (x_vec[1] ==  x_vec[1]) ?  x_vec[1] : x_vec_old[1];
//
//        error_x = (x_vec[0] == 0.) ? fabs(dx_vec[0]) : fabs(dx_vec[0]/x_vec[0]);
//
//
//        // if (iter < 4) {
//        //     std::cout << "error_x  " <<  error_x  << "  step" <<  iter <<'\n';
//        //     std::cout << "x  " <<  x_vec[0] << x_vec[1]  <<  x_vec[2]  <<'\n';
//        // }
//
//
//        if( (fabs(error_x) <= precision)  || residual_vec[0] <= precision_2){
//        iter_extra += 1;
//        keep_iteration = false;
//        }
//        else {
//        iter_extra = 0;
//        }
//
//        if( (iter_extra >= iter_extra_max) ) {
//        keep_iteration = false;
//        }
//        else if( iter >= iter_max) {
//        keep_iteration = false;
//        // DEBUG
//        // -------------------------
//        std::cout << "error_x  " <<  error_x  << "  ->  Newton-Rhapson did not converge !!!" << '\n';
//        std::cout << "FN dx[0]  " <<  dx_vec[0]  << "  step  "    <<  1.*iter <<'\n';
//        std::cout << "FN x  " <<  "  " << x_vec[0] <<  "  " << x_vec[1]  << "  " <<  x_vec[2]  <<'\n';
//        std::cout << "Orig x  " <<  "  " << x_vec_orig[0] <<  "  " << x_vec_orig[1]  << "  " <<  x_vec_orig[2]  <<'\n';
//
//        std::cout << " S2  " <<  "  " << S2 << "   chi  " << geo_vars.chi <<  "\n";
//
//        std::cout << " diff pressure  " <<  x_vec[0] - vars.D - vars.E <<   "   press: " <<  pressure
//        << " diff " <<  x_vec[0] - vars.D - vars.E  - pressure <<   "\n" << "\n";
//        }
//
//    } // end  keep_iteration
//
//
}


#endif /* PERFECTFLUID_IMPL_HPP_ */
