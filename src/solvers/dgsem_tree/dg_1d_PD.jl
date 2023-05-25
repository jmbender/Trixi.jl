#try to return production and destruction terms instead of rhs function; only for VolumeIntegralPureLGLFiniteVolume
#based on file "dg_1d.jl"
@muladd begin

#####################################################################################################
#unchanged

    # everything related to a DG semidiscretization in 1D,
    # currently limited to Lobatto-Legendre nodes
    
    # This method is called when a SemidiscretizationHyperbolic is constructed.
    # It constructs the basic `cache` used throughout the simulation to compute
    # the RHS etc.
    function create_cache(mesh::TreeMesh{1}, equations,
                          dg::DG, RealT, uEltype)
      # Get cells for which an element needs to be created (i.e. all leaf cells)
      leaf_cell_ids = local_leaf_cells(mesh.tree)
    
      elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)
    
      interfaces = init_interfaces(leaf_cell_ids, mesh, elements)
    
      boundaries = init_boundaries(leaf_cell_ids, mesh, elements)
    
      cache = (; elements, interfaces, boundaries)
    
      # Add specialized parts of the cache required to compute the volume integral etc.
      cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
    
      return cache
    end
    
    
    function create_cache(mesh::Union{TreeMesh{1}, StructuredMesh{1}, P4estMesh{1}}, equations,
                          volume_integral::VolumeIntegralPureLGLFiniteVolume, dg::DG, uEltype)
    
      A2dp1_x = Array{uEltype, 2}
      fstar1_L_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg)+1) for _ in 1:Threads.nthreads()]
      fstar1_R_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg)+1) for _ in 1:Threads.nthreads()]
    
      return (; fstar1_L_threaded, fstar1_R_threaded)
    end
    
    
    # TODO: Taal discuss/refactor timer, allowing users to pass a custom timer?
    
    function rhs!(du, u, t,
                  mesh::TreeMesh{1}, equations,
                  initial_condition, boundary_conditions, source_terms::Source,
                  dg::DG, cache) where {Source}
      # Reset du
      @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)
    
      # Calculate volume integral
      @trixi_timeit timer() "volume integral" calc_volume_integral!(
        du, u, mesh,
        have_nonconservative_terms(equations), equations,
        dg.volume_integral, dg, cache)
    
      # Prolong solution to interfaces
      @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
        cache, u, mesh, equations, dg.surface_integral, dg)
    
      # Calculate interface fluxes
      @trixi_timeit timer() "interface flux" calc_interface_flux!(
        cache.elements.surface_flux_values, mesh,
        have_nonconservative_terms(equations), equations,
        dg.surface_integral, dg, cache)
    
      # Prolong solution to boundaries
      @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
        cache, u, mesh, equations, dg.surface_integral, dg)
    
      # Calculate boundary fluxes
      @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
        cache, t, boundary_conditions, mesh,
        equations, dg.surface_integral, dg)
    
      # Calculate surface integrals
      @trixi_timeit timer() "surface integral" calc_surface_integral!(
        du, u, mesh, equations, dg.surface_integral, dg, cache)
    
      # Apply Jacobian from mapping to reference element
      @trixi_timeit timer() "Jacobian" apply_jacobian!(
        du, mesh, equations, dg, cache)
    
      # Calculate source terms
      @trixi_timeit timer() "source terms" calc_sources!(
        du, u, t, source_terms, equations, dg, cache)
    
      return nothing
    end
    
    
    # TODO: Taal dimension agnostic
    function calc_volume_integral!(du, u,
                                   mesh::TreeMesh{1},
                                   nonconservative_terms, equations,
                                   volume_integral::VolumeIntegralPureLGLFiniteVolume,
                                   dg::DGSEM, cache)
      @unpack volume_flux_fv = volume_integral
    
      # Calculate LGL FV volume integral
      @threaded for element in eachelement(dg, cache)
        fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
                   dg, cache, element, true)
      end
    
      return nothing
    end
    
    
    @inline function fv_kernel!(du, u,
                                mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                nonconservative_terms, equations,
                                volume_flux_fv, dg::DGSEM, cache, element, alpha=true)
      @unpack fstar1_L_threaded, fstar1_R_threaded = cache
      @unpack inverse_weights = dg.basis
    
      # Calculate FV two-point fluxes
      fstar1_L = fstar1_L_threaded[Threads.threadid()]
      fstar1_R = fstar1_R_threaded[Threads.threadid()]
      calcflux_fv!(fstar1_L, fstar1_R, u, mesh, nonconservative_terms, equations, volume_flux_fv,
                   dg, element, cache)
    
      # Calculate FV volume integral contribution
      for i in eachnode(dg)
        for v in eachvariable(equations)
          du[v, i, element] += ( alpha *
                                 (inverse_weights[i] * (fstar1_L[v, i+1] - fstar1_R[v, i])) )
    ###############safe here in production/destruction instead of du?
        end
      end
    
      return nothing
    end
    
    
    @inline function calcflux_fv!(fstar1_L, fstar1_R, u::AbstractArray{<:Any,3},
                                  mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                  nonconservative_terms::False,
                                  equations, volume_flux_fv, dg::DGSEM, element, cache)
    
      fstar1_L[:, 1           ] .= zero(eltype(fstar1_L))
      fstar1_L[:, nnodes(dg)+1] .= zero(eltype(fstar1_L))
      fstar1_R[:, 1           ] .= zero(eltype(fstar1_R))
      fstar1_R[:, nnodes(dg)+1] .= zero(eltype(fstar1_R))
    
      for i in 2:nnodes(dg)
        u_ll = get_node_vars(u, equations, dg, i-1, element)
        u_rr = get_node_vars(u, equations, dg, i  , element)
        flux = volume_flux_fv(u_ll, u_rr, 1, equations) # orientation 1: x direction
        set_node_vars!(fstar1_L, flux, equations, dg, i)
        set_node_vars!(fstar1_R, flux, equations, dg, i)
      end
    
      return nothing
    end
    
    
    
    # We pass the `surface_integral` argument solely for dispatch
    function prolong2interfaces!(cache, u,
                                 mesh::TreeMesh{1}, equations, surface_integral, dg::DG)
      @unpack interfaces = cache
    
      @threaded for interface in eachinterface(dg, cache)
        left_element  = interfaces.neighbor_ids[1, interface]
        right_element = interfaces.neighbor_ids[2, interface]
    
        # interface in x-direction
        for v in eachvariable(equations)
          interfaces.u[1, v, interface] = u[v, nnodes(dg), left_element]
          interfaces.u[2, v, interface] = u[v,          1, right_element]
        end
      end
    
      return nothing
    end
    
    function calc_interface_flux!(surface_flux_values,
                                  mesh::TreeMesh{1},
                                  nonconservative_terms::False, equations,
                                  surface_integral, dg::DG, cache)
      @unpack surface_flux = surface_integral
      @unpack u, neighbor_ids, orientations = cache.interfaces
    
      @threaded for interface in eachinterface(dg, cache)
        # Get neighboring elements
        left_id  = neighbor_ids[1, interface]
        right_id = neighbor_ids[2, interface]
    
        # Determine interface direction with respect to elements:
        # orientation = 1: left -> 2, right -> 1
        left_direction  = 2 * orientations[interface]
        right_direction = 2 * orientations[interface] - 1
    
        # Call pointwise Riemann solver
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, interface)
        flux = surface_flux(u_ll, u_rr, orientations[interface], equations)
    
        # Copy flux to left and right element storage
        for v in eachvariable(equations)
          surface_flux_values[v, left_direction,  left_id]  = flux[v]
          surface_flux_values[v, right_direction, right_id] = flux[v]
        end
      end
    end
    
    
    function prolong2boundaries!(cache, u,
                                 mesh::TreeMesh{1}, equations, surface_integral, dg::DG)
      @unpack boundaries = cache
      @unpack neighbor_sides = boundaries
    
      @threaded for boundary in eachboundary(dg, cache)
        element = boundaries.neighbor_ids[boundary]
    
        # boundary in x-direction
        if neighbor_sides[boundary] == 1
          # element in -x direction of boundary
          for v in eachvariable(equations)
            boundaries.u[1, v, boundary] = u[v, nnodes(dg), element]
          end
        else # Element in +x direction of boundary
          for v in eachvariable(equations)
            boundaries.u[2, v, boundary] = u[v, 1,          element]
          end
        end
      end
    
      return nothing
    end
    
    # TODO: Taal dimension agnostic
    function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                                 mesh::TreeMesh{1}, equations, surface_integral, dg::DG)
      @assert isempty(eachboundary(dg, cache))
    end
    
    
    function calc_boundary_flux!(cache, t, boundary_conditions::NamedTuple,
                                 mesh::TreeMesh{1}, equations, surface_integral, dg::DG)
      @unpack surface_flux_values = cache.elements
      @unpack n_boundaries_per_direction = cache.boundaries
    
      # Calculate indices
      lasts = accumulate(+, n_boundaries_per_direction)
      firsts = lasts - n_boundaries_per_direction .+ 1
    
      # Calc boundary fluxes in each direction
      calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[1],
                                       have_nonconservative_terms(equations), equations, surface_integral, dg, cache,
                                       1, firsts[1], lasts[1])
      calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[2],
                                       have_nonconservative_terms(equations), equations, surface_integral, dg, cache,
                                       2, firsts[2], lasts[2])
    end
    
    
    function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,3}, t,
                                              boundary_condition, nonconservative_terms::False, equations,
                                              surface_integral, dg::DG, cache,
                                              direction, first_boundary, last_boundary)
    
      @unpack surface_flux = surface_integral
      @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries
    
      @threaded for boundary in first_boundary:last_boundary
        # Get neighboring element
        neighbor = neighbor_ids[boundary]
    
        # Get boundary flux
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, boundary)
        if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
          u_inner = u_ll
        else # Element is on the right, boundary on the left
          u_inner = u_rr
        end
        x = get_node_coords(node_coordinates, equations, dg, boundary)
        flux = boundary_condition(u_inner, orientations[boundary], direction, x, t, surface_flux,
                                  equations)
    
        # Copy flux to left and right element storage
        for v in eachvariable(equations)
          surface_flux_values[v, direction, neighbor] = flux[v]
        end
      end
    
      return nothing
    end
    
   
    
    function calc_surface_integral!(du, u, mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                    equations, surface_integral, dg::DGSEM, cache)
      @unpack boundary_interpolation = dg.basis
      @unpack surface_flux_values = cache.elements
    
      # Note that all fluxes have been computed with outward-pointing normal vectors.
      # Access the factors only once before beginning the loop to increase performance.
      # We also use explicit assignments instead of `+=` to let `@muladd` turn these
      # into FMAs (see comment at the top of the file).
      factor_1 = boundary_interpolation[1,          1]
      factor_2 = boundary_interpolation[nnodes(dg), 2]
      @threaded for element in eachelement(dg, cache)
        ################split here again
        for v in eachvariable(equations)
          # surface at -x
          du[v, 1,          element] = (
            du[v, 1,          element] - surface_flux_values[v, 1, element] * factor_1)
    
          # surface at +x
          du[v, nnodes(dg), element] = (
            du[v, nnodes(dg), element] + surface_flux_values[v, 2, element] * factor_2)
        end
      end
    
      return nothing
    end
    
    
    function apply_jacobian!(du, mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                             equations, dg::DG, cache)
    
      @threaded for element in eachelement(dg, cache)
        factor = -cache.elements.inverse_jacobian[element]
    
        for i in eachnode(dg)
          for v in eachvariable(equations)
            du[v, i, element] *= factor
          end
        end
      end
    
      return nothing
    end
    
    
    # TODO: Taal dimension agnostic
    function calc_sources!(du, u, t, source_terms::Nothing,
                           equations::AbstractEquations{1}, dg::DG, cache)
      return nothing
    end
    
    function calc_sources!(du, u, t, source_terms,
                           equations::AbstractEquations{1}, dg::DG, cache)
    
      @threaded for element in eachelement(dg, cache)
        for i in eachnode(dg)
          u_local = get_node_vars(u, equations, dg, i, element)
          x_local = get_node_coords(cache.elements.node_coordinates, equations, dg, i, element)
          du_local = source_terms(u_local, x_local, t, equations)
          add_to_node_vars!(du, du_local, equations, dg, i, element)
        end
      end
    
      return nothing
    end
    
    
    end # @muladd
    