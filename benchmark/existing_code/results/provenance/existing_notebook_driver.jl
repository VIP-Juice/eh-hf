using DelimitedFiles
using Printf
using Random
using Serialization
using TOML


function argument_value(name::String, default::String)
    index = findfirst(==(name), ARGS)
    isnothing(index) && return default
    index == length(ARGS) && error("missing value after $name")
    return ARGS[index + 1]
end


output_dir = abspath(argument_value("--output-dir", "benchmark/results_existing_code"))
iterations = parse(Int, argument_value("--iterations", "300"))
time_limit = parse(Float64, argument_value("--time-limit", "300"))
tolerance = parse(Float64, argument_value("--tolerance", "1e-5"))
fine_nu = parse(Int, argument_value("--fine-nu", "120"))
fine_nv = parse(Int, argument_value("--fine-nv", "104"))
seed = parse(Int, argument_value("--seed", "20240904"))
mkpath(output_dir)


function localized_lattice_orbitals(width::Float64, centers)
    return ComplexF32[
        exp(-0.5 * width^2 * dot(k, k)) * cis(-dot(k, center))
        for k in kpoints, center in centers
    ]
end


Random.seed!(seed)
side = round(Int, sqrt(numbers[3]))
side^2 == numbers[3] || error("trion initializer requires a square number of holes")
Nk >= maximum(numbers) || error("momentum basis has $Nk states but needs at least $(maximum(numbers))")
centers = [T * [i, j] * L / side for i in 0:side-1 for j in 0:side-1]
electron_initial = localized_lattice_orbitals(5.5, centers)
hole_initial = localized_lattice_orbitals(4.0, centers)
C_mats_init = [copy(electron_initial), copy(electron_initial), hole_initial]
long_vec = slater_to_vec(C_mats_init)

# The notebook stores all working arrays as Float32.  Check its analytic
# gradient at the initial state, where the directional derivative is large
# enough to resolve reliably despite Float32 energy accumulation.
initial_gradient = similar(long_vec)
grad!(initial_gradient, long_vec)
direction = initial_gradient ./ norm(initial_gradient)
analytic_derivative = real(dot(initial_gradient, direction))
epsilon = 1.0f-2
finite_derivative = (
    Float64(energy!(long_vec .+ epsilon .* direction))
    - Float64(energy!(long_vec .- epsilon .* direction))
) / (2 * epsilon)
derivative_error = abs(analytic_derivative - finite_derivative)
derivative_relative_error = derivative_error / abs(analytic_derivative)

options = opt.Options(
    iterations=iterations,
    time_limit=time_limit,
    f_reltol=1e-10,
    g_abstol=tolerance,
    show_trace=true,
    show_every=10,
)
started = time()
solution = opt.optimize(energy!, grad!, long_vec, opt.ConjugateGradient(), options)
elapsed_seconds = time() - started

minimizer = opt.minimizer(solution)
final_C = vec_to_slater(minimizer)
final_energy = Float64(energy_calc!(final_C))
gradient = similar(minimizer)
grad!(gradient, minimizer)
maximum_gradient_component = max(maximum(abs, real.(gradient)), maximum(abs, imag.(gradient)))
energy_calc!(final_C)

function orthonormal_orbitals(coefficients)
    factor = cholesky(Hermitian(coefficients' * coefficients))
    return coefficients / factor.U
end

physical_orbitals = orthonormal_orbitals.(final_C)
fractional_points = [(u, v) for v in (0:fine_nv-1) ./ fine_nv for u in (0:fine_nu-1) ./ fine_nu]
positions = [L .* (T * [u, v]) for (u, v) in fractional_points]
phase = ComplexF64[cis(dot(k, r)) for r in positions, k in kpoints]
wavefunctions = [phase * coefficients / sqrt(A) for coefficients in physical_orbitals]
species_density = [vec(sum(abs2, wavefunction; dims=2)) for wavefunction in wavefunctions]
electron_density = species_density[1] + species_density[2]
hole_density = species_density[3]
area_weight = A / length(fractional_points)

open(joinpath(output_dir, "density.csv"), "w") do io
    println(io, "u,v,x,y,electron_density,hole_density")
    for index in eachindex(fractional_points)
        u, v = fractional_points[index]
        x, y = positions[index]
        println(io, join((u, v, x, y, electron_density[index], hole_density[index]), ','))
    end
end

checkpoint = Dict(
    "minimizer" => minimizer,
    "coefficient_matrices" => final_C,
    "physical_orbitals" => physical_orbitals,
)
serialize(joinpath(output_dir, "checkpoint.jls"), checkpoint)

metrics = Dict(
    "source" => Dict(
        "notebook" => get(ENV, "EH_HF_NOTEBOOK", "unknown"),
        "notebook_sha256" => get(ENV, "EH_HF_NOTEBOOK_SHA256", "unknown"),
        "hf_engine_cell_sha256" => get(ENV, "EH_HF_ENGINE_SHA256", "unknown"),
        "runner_sha256" => get(ENV, "EH_HF_RUNNER_SHA256", "unknown"),
        "driver_sha256" => get(ENV, "EH_HF_DRIVER_SHA256", "unknown"),
        "core_cells" => [0, 2, 3, 4, 6, 7],
    ),
    "parameters" => Dict(
        "n_electrons_up" => numbers[1],
        "n_electrons_down" => numbers[2],
        "n_holes" => numbers[3],
        "species" => species,
        "layers" => layers,
        "spin_locked" => spin_locked,
        "electron_mass_in_hole_units" => masses[1],
        "hole_mass_in_hole_units" => masses[3],
        "d_over_aB_h" => d,
        "hole_rs" => rsM,
        "momentum_cutoff_Rk" => Rk,
        "number_plane_waves" => Nk,
        "cell_matrix" => collect(eachrow(L .* T)),
        "area" => A,
    ),
    "result" => Dict(
        "converged" => maximum_gradient_component < tolerance,
        "optim_converged" => opt.converged(solution),
        "termination" => string(opt.summary(solution)),
        "iterations" => opt.iterations(solution),
        "elapsed_seconds" => elapsed_seconds,
        "total_energy_hartree_h" => final_energy,
        "energy_per_particle_hartree_h" => final_energy / sum(numbers),
        "paper_energy_per_particle_hartree_h" => -0.07652,
        "max_gradient_component" => maximum_gradient_component,
        "finite_difference_analytic_derivative" => analytic_derivative,
        "finite_difference_numeric_derivative" => finite_derivative,
        "finite_difference_epsilon" => epsilon,
        "finite_difference_derivative_error" => derivative_error,
        "finite_difference_relative_error" => derivative_relative_error,
        "integrated_electron_density" => area_weight * sum(electron_density),
        "integrated_hole_density" => area_weight * sum(hole_density),
        "max_electron_density" => maximum(electron_density),
        "max_hole_density" => maximum(hole_density),
    ),
)
open(joinpath(output_dir, "metrics.toml"), "w") do io
    TOML.print(io, metrics; sorted=true)
end

@printf("existing notebook engine: Nk=%d, iterations=%d, E/N=%+.8f, max|gradient component|=%.3e\n",
    Nk, opt.iterations(solution), final_energy / sum(numbers), maximum_gradient_component)
@printf("density integrals: Ne=%.10f, Nh=%.10f; derivative error=%.3e\n",
    area_weight * sum(electron_density), area_weight * sum(hole_density), derivative_error)
