### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 8b85bc47-3be7-4f59-a88a-3056887a06a1
using Dates

# ╔═╡ 2ea8bce2-88d8-11ef-0b73-2140d9176eeb
md"""
# FarmWorks Rossi Model
"""

# ╔═╡ 0f6a7534-6ff4-40d0-befd-d0f060a68779
md"""
This is an experimental notebook implementation of Rossi, 2007. The semi-ordered goals of this notebook are to,
1. Create a working, readable implementation of the Rossi model as it is documented in the 2007 paper. This is meant to serve as a prototype and learning exercise for future models.
2. Write documentation that explains both the code and paper at a granular level such that any seasoned developer could be onboarded to this notebook and begin making contributions in a relatively short amount of time.
3. Identify areas where uncertainty exists, or can be introduced, into the model. Begin considering how the model could be expressed in RxInfer with this uncertainty.
4. Identify technical limitations or challenges with existing software, find alternatives, or propose solutions (one I can think of: there does not currently exist a way to implement a piece-wise changepoint model in RxInfer. This may be problematic for our purposes, but it is possible we could get around such issues through clever use of distributed agents via RxEnvironments).
"""

# ╔═╡ 51236038-8d77-4984-a522-c72890cc4912
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 5541b153-cb98-4dd9-8429-dfc9aa426367
md"""

"""

# ╔═╡ f08decca-bb37-4b57-aba9-a7da16f8ce34
begin
	const SEASONAL_OOSPORE_DOSE = 1.0  			# SOD
	const HYDRO_THERMAL_TIME_MIN = 1.3 			# HT_min
	const HYDRO_THERMAL_TIME_MAX = 8.6 			# HT_max
	const OOSPORE_GERMINATION_MIN = 1.0 		# GER_min
	const ZOOSPORE_SURVIVAL_THRESHOLD = 1.0 	# SUZ_threshold
	const ZOOSPORE_INFECTION_THRESHOLD = 60.0 	# INF_threshold
end

# ╔═╡ 1c7d64e2-d9f7-4a30-a7c2-e59353ccce64
md"""
**TODO**: We need to determine the units for everything we are working with.
"""

# ╔═╡ be968b06-67ea-4e84-b755-6b71333c654c
struct Weather
    temperature::Vector{Float64}
    rainfall::Vector{Float64}
    relative_humidity::Vector{Float64}
    is_leaf_litter_moist::Vector{Bool}
end

# ╔═╡ 00c6f06a-ced8-4d62-a30e-b054767bde24
mutable struct OosporeCohort
    germination_level::Float64  # GER
	
    sporangia_survival::Float64  # SUS
    zoospore_survival::Float64  # SUZ
	
    are_zoospores_released::Bool  # REL
	
    zoospore_release_ratio::Float64  # ZRE ∈ [0, 1] 
    zoospore_dispersal_ratio::Float64  # ZDI ∈ [0, 1] 
    zoospore_infection_ratio::Float64  # ZIN ∈ [0, 1] 
	
    are_oil_spots_on_leaves::Bool  # OSL
	
    progress_of_dormancy_breaking_in_oospore_population::Float64 # DOR / hours
	
    germination_start_time::Int  # Time when cohort starts germination
end

# ╔═╡ 296df4f7-92ac-41f7-8a90-d275c65de8c2
md"""
**TODO**: We need to determine what these constants are, where they come from, and what they mean.
"""

# ╔═╡ 176ad92d-7455-44c7-924a-ba70ced20f19
md"""
**TODO**: What is hydro-thermal time? Read this: https://bsppjournals.onlinelibrary.wiley.com/doi/10.1111/j.1365-3059.2007.01738.x
"""

# ╔═╡ f1ea7a90-85d3-4d6e-a723-622d60e732de
md"""
**TODO**: Move all constant values into appropriately named and documented comments
"""

# ╔═╡ d6c4521c-a727-4a48-8ac2-36c4703bb11f
md"""
*Note*: All of these figures are uncertain estimates that could eventually be include in the generative model as priors and updated over time.
"""

# ╔═╡ 249a5d27-edb5-46af-b225-1ed43dc2eaf8
function calculate_hydro_thermal_time(temperature, is_leaf_litter_moist)
    if temperature > 0
        return is_leaf_litter_moist / (1330.1 - 116.19 * temperature + 2.6256 * temperature^2)
    else
        return 0.0
    end
end

# ╔═╡ ac37da59-ad5a-46ad-a7f7-195d27c60e9a
function calculate_oospore_dormancy_breaking(hydro_thermal_time)
    return exp(-15.891 * exp(-0.653 * (hydro_thermal_time + 1)))
end

# ╔═╡ f5da8168-a19d-41aa-8e54-39b2664c44f3
function calculate_sporangia_survival(temperature, relative_humidity)
    return 1.0 / (24 * (5.67 - 0.47 * temperature * (1 - relative_humidity / 100) + 0.01 * temperature * (1 - relative_humidity / 100)^2))
end

# ╔═╡ 9f58c27d-236a-4733-ba74-6b1ee603aaa9
function calculate_zoospore_survival(hours_after_release, wet_hours)
    return hours_after_release / wet_hours
end

# ╔═╡ 43cf0da1-e0a9-4b81-afb3-40ce5da849dc
function is_estimated_to_be_infected(wetness_duration, average_temperature_over_wetness_duration)
    return wetness_duration * average_temperature_over_wetness_duration >= ZOOSPORE_INFECTION_THRESHOLD
end

# ╔═╡ d44bd8ba-a24e-4a01-8b04-5a41c2936ae8
function run_model(weather::Weather, start_day::Date, end_day::Date)
    days = Dates.value(end_day - start_day)
    num_hours = days * 24
    hydro_thermal_time = 0.0  # Initialize hydro-thermal time
    cohorts = OosporeCohort[]  # List of cohorts

    for hour in 1:num_hours
        temperature = weather.temperature[hour]
        relative_humidity = weather.relative_humidity[hour]
        is_leaf_litter_moist = weather.is_leaf_litter_moist[hour]

        # Update HT (Hydro-thermal time)
        hydro_thermal_time += calculate_hydro_thermal_time(temperature, is_leaf_litter_moist)

        # Dormancy breaking (DOR) progress
        if hydro_thermal_time >= HYDRO_THERMAL_TIME_MIN && hydro_thermal_time <= HYDRO_THERMAL_TIME_MAX
			
            oospore_dormancy_breaking = calculate_oospore_dormancy_breaking(hydro_thermal_time)

            # Trigger new cohort on rain event
            if weather.rainfall[hour] >= 0.2
                push!(cohorts, OosporeCohort(0.0, 0.0, 0.0, false, 0.0, 0.0, 0.0, false, oospore_dormancy_breaking, hour))
            end
        end
	
		 # Update each cohort
		for cohort in cohorts
			# Germination progress
			if cohort.germination_level < OOSPORE_GERMINATION_MIN
				cohort.germination_level += calculate_hydro_thermal_time(temperature, is_leaf_litter_moist)
			end
	
			# Sporangia survival
			if cohort.germination_level >= OOSPORE_GERMINATION_MIN
				cohort.sporangia_survival += calculate_sporangia_survival(temperature, relative_humidity)
	
				# Zoospore release
				if cohort.sporangia_survival <= ZOOSPORE_SURVIVAL_THRESHOLD && is_leaf_litter_moist
					cohort.are_zoospores_released = true
					cohort.zoospore_release_ratio = cohort.germination_level
				end
			end
	
			# Zoospore dispersal and infection
			if cohort.are_zoospores_released && weather.rainfall[hour] >= 0.2
				cohort.zoospore_dispersal_ratio = cohort.zoospore_release_ratio
				if is_estimated_to_be_infected(cohort.sporangia_survival, temperature)
					cohort.zoospore_infection_ratio = cohort.
					cohort.are_oil_spots_on_leaves = true
				end
			end
		end
	end
    return cohorts
end

# ╔═╡ e7433753-825b-4390-ad22-d4c9e384ba31
md"""
We can make this nicer later.

Rossi, 2007: https://www.sciencedirect.com/science/article/pii/S0304380007005881?ref=cra_js_challenge&fr=RR-1
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "d7cd76e304b32b583eb96a7ac19153dc0f2d1730"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
"""

# ╔═╡ Cell order:
# ╟─2ea8bce2-88d8-11ef-0b73-2140d9176eeb
# ╟─0f6a7534-6ff4-40d0-befd-d0f060a68779
# ╟─51236038-8d77-4984-a522-c72890cc4912
# ╠═8b85bc47-3be7-4f59-a88a-3056887a06a1
# ╠═5541b153-cb98-4dd9-8429-dfc9aa426367
# ╠═f08decca-bb37-4b57-aba9-a7da16f8ce34
# ╟─1c7d64e2-d9f7-4a30-a7c2-e59353ccce64
# ╠═be968b06-67ea-4e84-b755-6b71333c654c
# ╠═00c6f06a-ced8-4d62-a30e-b054767bde24
# ╟─296df4f7-92ac-41f7-8a90-d275c65de8c2
# ╟─176ad92d-7455-44c7-924a-ba70ced20f19
# ╟─f1ea7a90-85d3-4d6e-a723-622d60e732de
# ╟─d6c4521c-a727-4a48-8ac2-36c4703bb11f
# ╠═249a5d27-edb5-46af-b225-1ed43dc2eaf8
# ╠═ac37da59-ad5a-46ad-a7f7-195d27c60e9a
# ╠═f5da8168-a19d-41aa-8e54-39b2664c44f3
# ╠═9f58c27d-236a-4733-ba74-6b1ee603aaa9
# ╠═43cf0da1-e0a9-4b81-afb3-40ce5da849dc
# ╠═d44bd8ba-a24e-4a01-8b04-5a41c2936ae8
# ╠═e7433753-825b-4390-ad22-d4c9e384ba31
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
