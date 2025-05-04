// Knowledge Graph Visualization using D3.js
class KnowledgeGraphVisualizer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.width = this.container.offsetWidth;
        this.height = 700;
        this.options = Object.assign({
            nodeRadius: 15,
            linkDistance: 180,
            chargeStrength: -400,
            colors: d3.scaleOrdinal(d3.schemeCategory10)
        }, options);
        
        this.svg = null;
        this.simulation = null;
        this.nodeElements = null;
        this.linkElements = null;
        this.textElements = null;
        
        // Store original data
        this.allNodes = [];
        this.allLinks = [];
        
        // Store filtered data
        this.nodes = [];
        this.links = [];
        
        // Track node types and relationship types for filtering
        this.nodeTypes = new Set();
        this.relationshipTypes = new Set();
        
        // Track highlighted elements
        this.highlightedNode = null;
        this.highlightedConnections = new Set();
        
        // Filtering state
        this.activeFilters = {
            nodeTypes: new Set(),
            relationshipTypes: new Set(),
            searchTerm: ""
        };
        
        // Create tooltip
        this.tooltipDiv = d3.select("body").append("div")
            .attr("class", "d3-tooltip")
            .style("opacity", 0);
        
        // Initialize visualization
        this.initSimulation();
        
        // Create progress indicator
        this.progressContainer = d3.select(`#${this.containerId}-progress-container`);
        if (!this.progressContainer.empty()) {
            this.setupProgressIndicator();
        }
    }
    
    setupProgressIndicator() {
        // Set up the progress indicator
        this.progressContainer.html(`
            <div class="progress-indicator">
                <div class="progress">
                    <div class="progress-bar progress-bar-striped progress-bar-animated bg-info" 
                         role="progressbar" style="width: 0%" 
                         aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                        0%
                    </div>
                </div>
                <div class="progress-status text-center mt-2">Ready</div>
            </div>
        `);
        
        this.progressBar = this.progressContainer.select(".progress-bar");
        this.progressStatus = this.progressContainer.select(".progress-status");
    }
    
    updateProgress(percent, status) {
        if (this.progressBar && !this.progressBar.empty()) {
            this.progressBar
                .style("width", `${percent}%`)
                .attr("aria-valuenow", percent)
                .text(`${percent}%`);
                
            // Update colors based on progress
            if (percent < 30) {
                this.progressBar.attr("class", "progress-bar progress-bar-striped progress-bar-animated bg-danger");
            } else if (percent < 70) {
                this.progressBar.attr("class", "progress-bar progress-bar-striped progress-bar-animated bg-warning");
            } else {
                this.progressBar.attr("class", "progress-bar progress-bar-striped progress-bar-animated bg-success");
            }
            
            if (status) {
                this.progressStatus.text(status);
            }
        }
    }
    
    initSimulation() {
        // Create SVG element
        this.svg = d3.select(`#${this.containerId}`)
            .append("svg")
            .attr("width", this.width)
            .attr("height", this.height)
            .attr("viewBox", [0, 0, this.width, this.height])
            .attr("class", "graph-svg");
        
        // Add zoom functionality
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                this.svg.selectAll("g").attr("transform", event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create main group element for zooming
        this.g = this.svg.append("g");
        
        // Initialize the force simulation
        this.simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(this.options.linkDistance))
            .force("charge", d3.forceManyBody().strength(this.options.chargeStrength))
            .force("center", d3.forceCenter(this.width / 2, this.height / 2))
            .on("tick", () => this.ticked());
        
        // Create arrow marker for links
        this.svg.append("defs").selectAll("marker")
            .data(["end"])
            .enter().append("marker")
            .attr("id", d => d)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", this.options.nodeRadius + 10)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");
        
        // Add legend and filter containers
        this.filterContainer = d3.select(`#graph-filters`);
        if (this.filterContainer.empty()) {
            this.filterContainer = d3.select(`#${this.containerId}-container`)
                .append("div")
                .attr("id", "graph-filters")
                .attr("class", "graph-filters mt-4");
        }
        
        this.legendContainer = d3.select(`#graph-legend`);
        if (this.legendContainer.empty()) {
            this.legendContainer = d3.select(`#${this.containerId}-container`)
                .append("div")
                .attr("id", "graph-legend")
                .attr("class", "graph-legend mt-4");
        }
    }
    
    loadData(graphId) {
        // Show loading state
        this.updateProgress(10, "Loading graph data...");
        this.container.innerHTML = `
            <div class="text-center p-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3">Loading knowledge graph data...</p>
            </div>
        `;
        
        // Load graph data from the server
        fetch(`/graph/${graphId}/data`)
            .then(response => {
                this.updateProgress(30, "Processing response...");
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                this.updateProgress(50, "Preparing visualization...");
                this.container.innerHTML = '';
                
                // Re-initialize the SVG after clearing the container
                this.initSimulation();
                
                // Process and render data
                this.processGraphData(data.nodes, data.relationships);
                this.updateProgress(100, "Graph loaded successfully");
            })
            .catch(error => {
                console.error('Error loading graph data:', error);
                this.displayError('Failed to load graph data. Please try again later.');
                this.updateProgress(100, "Error loading graph");
            });
    }
    
    processGraphData(nodes, links) {
        this.updateProgress(60, "Processing nodes and relationships...");
        
        // Process nodes and add properties
        this.allNodes = nodes.map((node, i) => {
            // Extract or infer node type from properties or labels
            const nodeType = node.type || 
                             node.labels?.[0] || 
                             (node.properties?.type) || 
                             'Entity';
            
            this.nodeTypes.add(nodeType);
            
            return {
                ...node,
                nodeType: nodeType,
                color: this.options.colors(nodeType.charCodeAt(0) % 10),
                index: i
            };
        });
        
        // Process links and add properties
        this.allLinks = links.map(link => {
            const relType = link.type || 'relationship';
            this.relationshipTypes.add(relType);
            
            return {
                source: link.source,
                target: link.target,
                type: relType,
                color: this.getLinkColor(relType)
            };
        });
        
        this.updateProgress(70, "Creating filter controls...");
        
        // Create filters
        this.createFilterControls();
        
        // Initial visualization with all data
        this.nodes = [...this.allNodes];
        this.links = [...this.allLinks];
        
        this.updateProgress(80, "Rendering graph...");
        this.renderGraph();
        
        this.updateProgress(100, "Complete");
    }
    
    createFilterControls() {
        if (this.filterContainer.empty()) return;
        
        this.filterContainer.html(`
            <div class="row mb-3">
                <div class="col-12">
                    <h5>Filter Graph</h5>
                </div>
            </div>
            
            <div class="row g-3">
                <!-- Search input -->
                <div class="col-lg-4">
                    <div class="input-group">
                        <input type="text" id="graph-search" class="form-control" 
                               placeholder="Search nodes..." aria-label="Search nodes">
                        <button class="btn btn-outline-secondary" type="button" id="graph-search-clear">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
                
                <!-- Node type filter -->
                <div class="col-lg-4">
                    <select id="node-type-filter" class="form-select" aria-label="Filter by node type">
                        <option value="">All node types</option>
                    </select>
                </div>
                
                <!-- Relationship type filter -->
                <div class="col-lg-4">
                    <select id="relationship-type-filter" class="form-select" aria-label="Filter by relationship type">
                        <option value="">All relationship types</option>
                    </select>
                </div>
                
                <!-- Density slider -->
                <div class="col-12">
                    <label for="graph-density" class="form-label">Graph Density</label>
                    <input type="range" class="form-range" min="10" max="100" value="100" id="graph-density">
                </div>
            </div>
        `);
        
        // Add options to filter selects
        const nodeTypeSelect = this.filterContainer.select("#node-type-filter");
        [...this.nodeTypes].sort().forEach(type => {
            nodeTypeSelect.append("option")
                .attr("value", type)
                .text(type);
        });
        
        const relationshipTypeSelect = this.filterContainer.select("#relationship-type-filter");
        [...this.relationshipTypes].sort().forEach(type => {
            relationshipTypeSelect.append("option")
                .attr("value", type)
                .text(type);
        });
        
        // Add event listeners for filters
        this.filterContainer.select("#graph-search").on("input", (event) => {
            const searchTerm = event.target.value.toLowerCase().trim();
            this.activeFilters.searchTerm = searchTerm;
            this.applyFilters();
        });
        
        this.filterContainer.select("#graph-search-clear").on("click", () => {
            this.filterContainer.select("#graph-search").property("value", "");
            this.activeFilters.searchTerm = "";
            this.applyFilters();
        });
        
        nodeTypeSelect.on("change", (event) => {
            const selectedType = event.target.value;
            if (selectedType) {
                this.activeFilters.nodeTypes = new Set([selectedType]);
            } else {
                this.activeFilters.nodeTypes = new Set();
            }
            this.applyFilters();
        });
        
        relationshipTypeSelect.on("change", (event) => {
            const selectedType = event.target.value;
            if (selectedType) {
                this.activeFilters.relationshipTypes = new Set([selectedType]);
            } else {
                this.activeFilters.relationshipTypes = new Set();
            }
            this.applyFilters();
        });
        
        this.filterContainer.select("#graph-density").on("input", (event) => {
            const density = parseInt(event.target.value) / 100;
            this.updateGraphDensity(density);
        });
    }
    
    applyFilters() {
        // Filter nodes based on search and node type filters
        this.nodes = this.allNodes.filter(node => {
            // Filter by search term
            const matchesSearch = !this.activeFilters.searchTerm || 
                                 node.label.toLowerCase().includes(this.activeFilters.searchTerm);
            
            // Filter by node type
            const matchesType = this.activeFilters.nodeTypes.size === 0 || 
                               this.activeFilters.nodeTypes.has(node.nodeType);
            
            return matchesSearch && matchesType;
        });
        
        // Get set of visible node IDs for link filtering
        const visibleNodeIds = new Set(this.nodes.map(n => n.id));
        
        // Filter links based on relationship type and visible nodes
        this.links = this.allLinks.filter(link => {
            // Filter by relationship type
            const matchesType = this.activeFilters.relationshipTypes.size === 0 || 
                               this.activeFilters.relationshipTypes.has(link.type);
            
            // Only show links between visible nodes
            const nodesVisible = visibleNodeIds.has(link.source.id || link.source) && 
                                visibleNodeIds.has(link.target.id || link.target);
            
            return matchesType && nodesVisible;
        });
        
        // Update the visualization
        this.renderGraph();
    }
    
    updateGraphDensity(density) {
        // Calculate how many nodes to show based on density
        const nodeCount = Math.max(1, Math.floor(this.allNodes.length * density));
        const visibleNodes = this.allNodes.slice(0, nodeCount);
        
        // Update nodes
        this.nodes = visibleNodes;
        
        // Get set of visible node IDs
        const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
        
        // Filter links to only show those connecting visible nodes
        this.links = this.allLinks.filter(link => 
            visibleNodeIds.has(link.source.id || link.source) && 
            visibleNodeIds.has(link.target.id || link.target)
        );
        
        // Update visualization
        this.renderGraph();
    }
    
    getLinkColor(relationType) {
        // Generate consistent colors for relationship types
        const typeIndex = [...this.relationshipTypes].sort().indexOf(relationType);
        const colors = [
            "#666", "#888", "#aaa", "#999", "#777", 
            "#555", "#444", "#333", "#222", "#111"
        ];
        return colors[typeIndex % colors.length];
    }
    
    renderGraph() {
        // Create relationships first (so they appear behind nodes)
        this.linkElements = this.g.selectAll(".link")
            .data(this.links, d => `${d.source.id || d.source}-${d.target.id || d.target}-${d.type}`)
            .join(
                enter => enter.append("line")
                    .attr("class", "link")
                    .attr("stroke", d => d.color || "#999")
                    .attr("stroke-opacity", 0.6)
                    .attr("stroke-width", 2)
                    .attr("marker-end", "url(#end)")
                    .on("mouseover", (event, d) => this.handleLinkMouseOver(event, d))
                    .on("mouseout", () => this.handleLinkMouseOut()),
                update => update
                    .attr("stroke", d => d.color || "#999")
                    .attr("stroke-opacity", 0.6),
                exit => exit.remove()
            );
            
        // Create relationship labels
        this.linkLabelElements = this.g.selectAll(".link-label")
            .data(this.links, d => `${d.source.id || d.source}-${d.target.id || d.target}-${d.type}`)
            .join(
                enter => enter.append("text")
                    .attr("class", "link-label")
                    .attr("dy", -5)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#777")
                    .attr("font-size", "11px")
                    .attr("pointer-events", "none")
                    .text(d => d.type),
                update => update.text(d => d.type),
                exit => exit.remove()
            );
        
        // Create nodes
        this.nodeElements = this.g.selectAll(".node")
            .data(this.nodes, d => d.id)
            .join(
                enter => enter.append("circle")
                    .attr("class", d => `node node-type-${d.nodeType.replace(/\s+/g, '-').toLowerCase()}`)
                    .attr("r", this.options.nodeRadius)
                    .attr("fill", d => d.color)
                    .attr("stroke", "#fff")
                    .attr("stroke-width", 2)
                    .call(this.setupDragHandlers())
                    .on("mouseover", (event, d) => this.handleNodeMouseOver(event, d))
                    .on("mouseout", () => this.handleNodeMouseOut())
                    .on("click", (event, d) => this.handleNodeClick(event, d)),
                update => update
                    .attr("class", d => `node node-type-${d.nodeType.replace(/\s+/g, '-').toLowerCase()}`)
                    .attr("fill", d => d.color),
                exit => exit.remove()
            );
        
        // Create node labels
        this.textElements = this.g.selectAll(".node-label")
            .data(this.nodes, d => d.id)
            .join(
                enter => enter.append("text")
                    .attr("class", "node-label")
                    .attr("dy", 4)
                    .attr("text-anchor", "middle")
                    .attr("fill", d => this.getLabelColor(d.color))
                    .attr("font-size", "11px")
                    .attr("pointer-events", "none")
                    .text(d => this.truncateLabel(d.label)),
                update => update
                    .attr("fill", d => this.getLabelColor(d.color))
                    .text(d => this.truncateLabel(d.label)),
                exit => exit.remove()
            );
        
        // Update simulation
        this.simulation.nodes(this.nodes);
        this.simulation.force("link").links(this.links);
        this.simulation.alpha(1).restart();
        
        // Update legend
        this.updateLegend();
    }
    
    getLabelColor(backgroundColor) {
        // Determine if text should be white or black based on background color
        const rgb = d3.rgb(backgroundColor);
        const brightness = (rgb.r * 299 + rgb.g * 587 + rgb.b * 114) / 1000;
        return brightness > 125 ? "#000" : "#fff";
    }
    
    truncateLabel(label, maxLength = 10) {
        return label?.length > maxLength ? label.slice(0, maxLength) + '...' : (label || 'n/a');
    }
    
    ticked() {
        // Update node positions
        this.nodeElements
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
        
        // Update text positions
        this.textElements
            .attr("x", d => d.x)
            .attr("y", d => d.y);
        
        // Update link positions
        this.linkElements
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
            
        // Update link label positions
        this.linkLabelElements
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);
    }
    
    setupDragHandlers() {
        return d3.drag()
            .on("start", (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on("drag", (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on("end", (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }
    
    handleNodeMouseOver(event, d) {
        // Show tooltip with full label text and properties
        this.tooltipDiv.transition()
            .duration(200)
            .style("opacity", .9);
        
        let tooltipContent = `<strong>${d.label}</strong><br/>`;
        tooltipContent += `<span class="badge bg-secondary">${d.nodeType}</span><br/>`;
        
        // Add properties if available
        if (d.properties) {
            tooltipContent += `<hr class="my-1"/>`;
            Object.entries(d.properties).forEach(([key, value]) => {
                if (key !== 'id' && key !== 'label' && key !== 'type') {
                    tooltipContent += `<small>${key}: ${value}</small><br/>`;
                }
            });
        }
        
        this.tooltipDiv.html(tooltipContent)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
            
        // Highlight node
        d3.select(event.currentTarget)
            .attr("stroke", "#ff0")
            .attr("stroke-width", 3);
    }
    
    handleNodeMouseOut() {
        // Hide tooltip
        this.tooltipDiv.transition()
            .duration(500)
            .style("opacity", 0);
            
        // Remove highlight if not clicked
        if (!this.highlightedNode) {
            this.nodeElements
                .attr("stroke", "#fff")
                .attr("stroke-width", 2);
                
            this.linkElements
                .attr("stroke-opacity", 0.6)
                .attr("stroke-width", 2);
        }
    }
    
    handleLinkMouseOver(event, d) {
        // Show tooltip for relationship
        this.tooltipDiv.transition()
            .duration(200)
            .style("opacity", .9);
            
        const sourceLabel = d.source.label || d.source;
        const targetLabel = d.target.label || d.target;
        
        this.tooltipDiv.html(`<strong>${sourceLabel}</strong> ${d.type} <strong>${targetLabel}</strong>`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
            
        // Highlight link
        d3.select(event.currentTarget)
            .attr("stroke-width", 4)
            .attr("stroke-opacity", 1);
    }
    
    handleLinkMouseOut() {
        // Hide tooltip
        this.tooltipDiv.transition()
            .duration(500)
            .style("opacity", 0);
            
        // Remove highlight if no node is highlighted
        if (!this.highlightedNode) {
            this.linkElements
                .attr("stroke-opacity", 0.6)
                .attr("stroke-width", 2);
        }
    }
    
    handleNodeClick(event, d) {
        event.preventDefault();
        
        // If already highlighted, reset the highlight
        if (this.highlightedNode === d.id) {
            this.resetHighlight();
            return;
        }
        
        // Reset any existing highlights
        this.resetHighlight();
        
        // Set this node as highlighted
        this.highlightedNode = d.id;
        
        // Find connected nodes and links
        this.highlightedConnections = new Set();
        
        this.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;
            
            if (sourceId === d.id || targetId === d.id) {
                // Add this link to highlighted set
                this.highlightedConnections.add(`${sourceId}-${targetId}-${link.type}`);
                
                // Add connected node to highlighted set
                if (sourceId === d.id) {
                    this.highlightedConnections.add(targetId);
                } else {
                    this.highlightedConnections.add(sourceId);
                }
            }
        });
        
        // Apply highlighting
        this.nodeElements
            .attr("opacity", node => 
                node.id === d.id || this.highlightedConnections.has(node.id) ? 1 : 0.2
            )
            .attr("stroke", node => 
                node.id === d.id ? "#ff0" : (this.highlightedConnections.has(node.id) ? "#ffa500" : "#fff")
            )
            .attr("stroke-width", node => 
                node.id === d.id ? 3 : (this.highlightedConnections.has(node.id) ? 2.5 : 2)
            );
            
        this.textElements
            .attr("opacity", node => 
                node.id === d.id || this.highlightedConnections.has(node.id) ? 1 : 0.2
            );
            
        this.linkElements
            .attr("opacity", link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                const linkId = `${sourceId}-${targetId}-${link.type}`;
                return this.highlightedConnections.has(linkId) ? 1 : 0.1;
            })
            .attr("stroke-width", link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                const linkId = `${sourceId}-${targetId}-${link.type}`;
                return this.highlightedConnections.has(linkId) ? 3 : 1.5;
            });
            
        this.linkLabelElements
            .attr("opacity", link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                const linkId = `${sourceId}-${targetId}-${link.type}`;
                return this.highlightedConnections.has(linkId) ? 1 : 0.1;
            });
    }
    
    resetHighlight() {
        this.highlightedNode = null;
        this.highlightedConnections.clear();
        
        // Reset node appearance
        this.nodeElements
            .attr("opacity", 1)
            .attr("stroke", "#fff")
            .attr("stroke-width", 2);
            
        // Reset text appearance
        this.textElements
            .attr("opacity", 1);
            
        // Reset link appearance
        this.linkElements
            .attr("opacity", 1)
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", 2);
            
        // Reset link label appearance
        this.linkLabelElements
            .attr("opacity", 1);
    }
    
    updateLegend() {
        if (this.legendContainer.empty()) return;
        
        // Get unique node types and relationship types
        const nodeTypes = [...this.nodeTypes].sort();
        const relationshipTypes = [...this.relationshipTypes].sort();
        
        // Create legend
        this.legendContainer.html(`
            <div class="row">
                <div class="col-md-6">
                    <h5>Node Types</h5>
                    <div class="node-type-legend"></div>
                </div>
                <div class="col-md-6">
                    <h5>Relationship Types</h5>
                    <div class="relationship-type-legend"></div>
                </div>
            </div>
        `);
        
        // Add node type legend items
        const nodeTypeLegend = this.legendContainer.select(".node-type-legend");
        
        nodeTypes.forEach((type, i) => {
            const color = this.options.colors(type.charCodeAt(0) % 10);
            
            nodeTypeLegend.append("div")
                .attr("class", "legend-item d-flex align-items-center mb-2")
                .html(`
                    <span class="legend-color me-2" style="background: ${color}; width: 15px; height: 15px; border-radius: 50%;"></span>
                    <span class="legend-label">${type}</span>
                    <span class="ms-auto badge bg-secondary">${this.countNodesByType(type)}</span>
                `);
        });
        
        // Add relationship type legend items
        const relationshipTypeLegend = this.legendContainer.select(".relationship-type-legend");
        
        relationshipTypes.forEach((type, i) => {
            const color = this.getLinkColor(type);
            
            relationshipTypeLegend.append("div")
                .attr("class", "legend-item d-flex align-items-center mb-2")
                .html(`
                    <span class="legend-color me-2" style="background: ${color}; width: 25px; height: 3px;"></span>
                    <span class="legend-label">${type}</span>
                    <span class="ms-auto badge bg-secondary">${this.countLinksByType(type)}</span>
                `);
        });
    }
    
    countNodesByType(type) {
        return this.nodes.filter(node => node.nodeType === type).length;
    }
    
    countLinksByType(type) {
        return this.links.filter(link => link.type === type).length;
    }
    
    displayError(message) {
        this.container.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <h5><i class="fas fa-exclamation-triangle me-2"></i> Error</h5>
                <p>${message}</p>
            </div>
        `;
    }
    
    resize() {
        this.width = this.container.offsetWidth;
        
        this.svg
            .attr("width", this.width)
            .attr("viewBox", [0, 0, this.width, this.height]);
        
        this.simulation.force("center", d3.forceCenter(this.width / 2, this.height / 2));
        this.simulation.restart();
    }
    
    // Function to search for a specific entity and highlight it
    searchEntity(term) {
        if (!term) {
            this.resetHighlight();
            return false;
        }
        
        const lowerTerm = term.toLowerCase();
        const matchedNode = this.nodes.find(node => 
            node.label.toLowerCase().includes(lowerTerm)
        );
        
        if (matchedNode) {
            this.handleNodeClick({ preventDefault: () => {} }, matchedNode);
            
            // Center the view on this node
            const svg = this.svg.node();
            const currentZoom = d3.zoomTransform(svg);
            
            d3.select(svg)
                .transition()
                .duration(750)
                .call(d3.zoom().transform, 
                    d3.zoomIdentity
                        .translate(this.width / 2, this.height / 2)
                        .scale(currentZoom.k)
                        .translate(-matchedNode.x, -matchedNode.y));
                        
            return true;
        }
        
        return false;
    }
}
