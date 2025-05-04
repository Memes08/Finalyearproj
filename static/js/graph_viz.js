// Knowledge Graph Visualization using D3.js
class KnowledgeGraphVisualizer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.width = this.container.offsetWidth;
        this.height = 700;
        this.options = Object.assign({
            nodeRadius: 12,
            linkDistance: 150,
            chargeStrength: -300,
            colors: d3.scaleOrdinal(d3.schemeCategory10)
        }, options);
        
        // Visualization elements
        this.svg = null;
        this.simulation = null;
        this.nodeElements = null;
        this.linkElements = null;
        this.textElements = null;
        this.allNodes = []; // Store all nodes for filtering
        this.allLinks = []; // Store all links for filtering
        this.nodes = [];    // Currently visible nodes
        this.links = [];    // Currently visible links
        this.selectedNode = null; // Track currently selected node
        this.nodeCategories = new Set(); // Store node categories
        
        // Node visibility tracking
        this.hiddenNodes = new Set();
        this.hiddenLinks = new Set();
        this.filterActive = false;
        
        // Create tooltip
        this.tooltipDiv = d3.select("body").append("div")
            .attr("class", "d3-tooltip")
            .style("opacity", 0);
        
        this.initSimulation();
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
            .force("collision", d3.forceCollide().radius(this.options.nodeRadius * 1.5))
            .alphaDecay(0.028) // Slower decay for better stabilization
            .on("tick", () => this.ticked());
        
        // Create arrow marker for links
        this.svg.append("defs").selectAll("marker")
            .data(["end"])
            .enter().append("marker")
            .attr("id", d => d)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 25)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");
        
        // Add legend
        this.legendContainer = d3.select(`#${this.containerId}`)
            .append("div")
            .attr("class", "graph-legend");
        
        // Subscribe to custom events
        document.addEventListener('graphSearchEvent', (e) => this.handleSearch(e.detail.searchTerm));
        document.addEventListener('graphResetEvent', () => this.resetFilters());
        document.addEventListener('graphLayoutEvent', (e) => this.changeLayout(e.detail.layout));
        document.addEventListener('graphCategoryFilterEvent', (e) => this.filterByCategory(e.detail.category));
        document.addEventListener('graphRelationshipFilterEvent', (e) => this.filterByRelationship(e.detail.relationship));
    }
    
    loadData(graphId) {
        // Load graph data from the server
        fetch(`/graph/${graphId}/data`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Store original data for filtering
                this.allNodes = data.nodes.map(node => ({
                    ...node,
                    category: this.extractCategory(node.label)
                }));
                
                this.allLinks = data.relationships.map(l => ({
                    source: l.source,
                    target: l.target,
                    type: l.type
                }));
                
                // Extract categories for filtering
                this.allNodes.forEach(node => {
                    this.nodeCategories.add(node.category);
                });
                
                // Populate the category filter dropdown
                this.updateCategoryFilter(Array.from(this.nodeCategories));
                
                // Populate relationship filter
                const relationshipTypes = [...new Set(this.allLinks.map(l => l.type))];
                this.updateRelationshipFilter(relationshipTypes);
                
                // Show the whole graph initially
                this.updateGraph(this.allNodes, this.allLinks);
                
                // Dispatch event to update UI counters
                document.dispatchEvent(new CustomEvent('graphStatsUpdated', {
                    detail: {
                        nodeCount: this.allNodes.length,
                        relationshipCount: this.allLinks.length
                    }
                }));
            })
            .catch(error => {
                console.error('Error loading graph data:', error);
                this.displayError('Failed to load graph data. Please try again later.');
            });
    }
    
    extractCategory(label) {
        // Extract a general category from a node label
        // This is a simple heuristic - can be improved for specific domains
        if (label.includes("Person") || label.includes("Actor") || label.includes("Director")) {
            return "Person";
        } else if (label.includes("Movie") || label.includes("Film")) {
            return "Movie";
        } else if (label.includes("Company") || label.includes("Studio")) {
            return "Organization";
        } else if (label.includes("Genre")) {
            return "Genre";
        } else {
            return "Other";
        }
    }
    
    updateGraph(nodes, links) {
        this.nodes = nodes;
        this.links = links;
        
        // Create relationships first (so they appear behind nodes)
        this.linkElements = this.g.selectAll(".link")
            .data(this.links)
            .join(
                enter => enter.append("line")
                    .attr("class", "link")
                    .attr("stroke", "#999")
                    .attr("stroke-opacity", 0.6)
                    .attr("stroke-width", 1.5)
                    .attr("data-source", d => d.source.id || d.source)
                    .attr("data-target", d => d.target.id || d.target)
                    .attr("data-type", d => d.type)
                    .attr("marker-end", "url(#end)")
                    .on("mouseover", (event, d) => this.handleLinkMouseOver(event, d))
                    .on("mouseout", () => this.handleMouseOut()),
                update => update,
                exit => exit.remove()
            );
            
        // Create relationship labels
        this.linkLabelElements = this.g.selectAll(".link-label")
            .data(this.links)
            .join(
                enter => enter.append("text")
                    .attr("class", "link-label")
                    .attr("dy", -5)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#666")
                    .attr("font-size", "10px")
                    .text(d => d.type),
                update => update.text(d => d.type),
                exit => exit.remove()
            );
        
        // Create nodes
        this.nodeElements = this.g.selectAll(".node")
            .data(this.nodes)
            .join(
                enter => enter.append("circle")
                    .attr("class", d => `node node-category-${d.category}`)
                    .attr("r", this.options.nodeRadius)
                    .attr("fill", d => this.getCategoryColor(d.category))
                    .attr("stroke", "#fff")
                    .attr("stroke-width", 1.5)
                    .attr("data-id", d => d.id)
                    .attr("data-label", d => d.label)
                    .attr("data-category", d => d.category)
                    .call(this.setupDragHandlers())
                    .on("mouseover", (event, d) => this.handleNodeMouseOver(event, d))
                    .on("mouseout", () => this.handleMouseOut())
                    .on("click", (event, d) => this.handleNodeClick(event, d)),
                update => update,
                exit => exit.remove()
            );
        
        // Create node labels
        this.textElements = this.g.selectAll(".node-label")
            .data(this.nodes)
            .join(
                enter => enter.append("text")
                    .attr("class", "node-label")
                    .attr("dy", 4)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#fff")
                    .attr("font-size", "10px")
                    .attr("pointer-events", "none")
                    .text(d => this.truncateLabel(d.label)),
                update => update.text(d => this.truncateLabel(d.label)),
                exit => exit.remove()
            );
        
        // Update simulation
        this.simulation.nodes(this.nodes);
        this.simulation.force("link").links(this.links);
        this.simulation.alpha(1).restart();
        
        // Update legend
        this.updateLegend();
    }
    
    getCategoryColor(category) {
        // Consistent colors for node categories
        const categoryColors = {
            'Person': '#ff7f0e',      // Orange
            'Movie': '#1f77b4',       // Blue
            'Organization': '#2ca02c', // Green
            'Genre': '#d62728',       // Red
            'Other': '#9467bd'        // Purple
        };
        return categoryColors[category] || '#9467bd';
    }
    
    truncateLabel(label, maxLength = 10) {
        return label.length > maxLength ? label.slice(0, maxLength) + '...' : label;
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
        // Show tooltip with full label text
        this.tooltipDiv.transition()
            .duration(200)
            .style("opacity", .9);
        
        this.tooltipDiv.html(`
            <strong>${d.label}</strong><br>
            <span class="text-muted">Category: ${d.category}</span>
        `)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
        
        // Highlight connected nodes and links
        this.highlightConnections(d);
    }
    
    handleLinkMouseOver(event, d) {
        // Show tooltip with relationship details
        this.tooltipDiv.transition()
            .duration(200)
            .style("opacity", .9);
        
        const sourceLabel = typeof d.source === 'object' ? d.source.label : 'Unknown';
        const targetLabel = typeof d.target === 'object' ? d.target.label : 'Unknown';
        
        this.tooltipDiv.html(`
            <strong>${d.type}</strong><br>
            <span>From: ${this.truncateLabel(sourceLabel, 20)}</span><br>
            <span>To: ${this.truncateLabel(targetLabel, 20)}</span>
        `)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
    }
    
    handleMouseOut() {
        // Hide tooltip
        this.tooltipDiv.transition()
            .duration(500)
            .style("opacity", 0);
        
        // Reset highlighting if no node is selected
        if (!this.selectedNode) {
            this.resetHighlighting();
        }
    }
    
    handleNodeClick(event, d) {
        // Toggle node selection
        if (this.selectedNode === d) {
            // Deselect current node
            this.selectedNode = null;
            this.resetHighlighting();
            this.resetFilters(); // Show all nodes
            
            // Dispatch event for UI updates
            document.dispatchEvent(new CustomEvent('nodeDeselected'));
        } else {
            // Select new node
            this.selectedNode = d;
            this.filterToNodeAndConnections(d);
            
            // Dispatch event for UI updates
            document.dispatchEvent(new CustomEvent('nodeSelected', {
                detail: {
                    node: d,
                    connections: this.getNodeConnections(d)
                }
            }));
        }
    }
    
    highlightConnections(node) {
        // Fade all nodes and links
        this.nodeElements.attr("opacity", 0.3);
        this.linkElements.attr("opacity", 0.1);
        this.textElements.attr("opacity", 0.3);
        this.linkLabelElements.attr("opacity", 0);
        
        // Get connected nodes and links
        const connectedNodeIds = new Set();
        connectedNodeIds.add(node.id);
        
        // Find connected links and nodes
        const connectedLinks = this.links.filter(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            
            if (sourceId === node.id || targetId === node.id) {
                // Add connected node to the set
                if (sourceId === node.id) connectedNodeIds.add(targetId);
                if (targetId === node.id) connectedNodeIds.add(sourceId);
                return true;
            }
            return false;
        });
        
        // Highlight the node itself
        this.nodeElements
            .filter(d => connectedNodeIds.has(d.id))
            .attr("opacity", 1);
        
        // Highlight connected links
        this.linkElements
            .filter(link => {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                return sourceId === node.id || targetId === node.id;
            })
            .attr("opacity", 1)
            .attr("stroke-width", 2);
        
        // Highlight link labels
        this.linkLabelElements
            .filter(link => {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                return sourceId === node.id || targetId === node.id;
            })
            .attr("opacity", 1);
        
        // Highlight connected node labels
        this.textElements
            .filter(d => connectedNodeIds.has(d.id))
            .attr("opacity", 1);
    }
    
    resetHighlighting() {
        // Reset opacity for all elements
        this.nodeElements.attr("opacity", 1);
        this.linkElements.attr("opacity", 0.6).attr("stroke-width", 1.5);
        this.textElements.attr("opacity", 1);
        this.linkLabelElements.attr("opacity", d => d.highlighted ? 1 : 0);
    }
    
    getNodeConnections(node) {
        // Get all connections for a node
        const connections = {
            incoming: [],
            outgoing: []
        };
        
        this.links.forEach(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            
            if (sourceId === node.id) {
                // Outgoing connection
                const targetNode = this.nodes.find(n => n.id === targetId);
                if (targetNode) {
                    connections.outgoing.push({
                        node: targetNode,
                        relationship: link.type
                    });
                }
            }
            
            if (targetId === node.id) {
                // Incoming connection
                const sourceNode = this.nodes.find(n => n.id === sourceId);
                if (sourceNode) {
                    connections.incoming.push({
                        node: sourceNode,
                        relationship: link.type
                    });
                }
            }
        });
        
        return connections;
    }
    
    filterToNodeAndConnections(node) {
        // Filter graph to show only the selected node and its connections
        const connectedNodeIds = new Set();
        connectedNodeIds.add(node.id);
        
        // Find connected nodes
        this.links.forEach(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            
            if (sourceId === node.id) connectedNodeIds.add(targetId);
            if (targetId === node.id) connectedNodeIds.add(sourceId);
        });
        
        // Filter nodes and links
        const filteredNodes = this.allNodes.filter(n => connectedNodeIds.has(n.id));
        const filteredLinks = this.allLinks.filter(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return connectedNodeIds.has(sourceId) && connectedNodeIds.has(targetId);
        });
        
        // Update graph with filtered data
        this.updateGraph(filteredNodes, filteredLinks);
        this.filterActive = true;
        
        // Center view on the selected node
        this.centerOnNode(node);
    }
    
    centerOnNode(node) {
        // Center the view on the selected node
        const svgElement = document.querySelector(`#${this.containerId} svg`);
        const transform = d3.zoomIdentity
            .translate(this.width / 2, this.height / 2)
            .scale(1.5)
            .translate(-node.x, -node.y);
        
        d3.select(svgElement)
            .transition()
            .duration(750)
            .call(d3.zoom().transform, transform);
    }
    
    resetFilters() {
        // Reset to show all nodes and links
        if (this.filterActive) {
            this.updateGraph(this.allNodes, this.allLinks);
            this.filterActive = false;
            this.selectedNode = null;
            
            // Reset zoom
            const svgElement = document.querySelector(`#${this.containerId} svg`);
            d3.select(svgElement)
                .transition()
                .duration(750)
                .call(d3.zoom().transform, d3.zoomIdentity);
            
            // Dispatch deselect event
            document.dispatchEvent(new CustomEvent('nodeDeselected'));
        }
    }
    
    handleSearch(searchTerm) {
        if (!searchTerm) {
            // If search is cleared, show all nodes
            this.resetFilters();
            return;
        }
        
        searchTerm = searchTerm.toLowerCase();
        
        // Find matching nodes
        const matchingNodes = this.allNodes.filter(node => 
            node.label.toLowerCase().includes(searchTerm)
        );
        
        if (matchingNodes.length === 0) {
            // No matches found
            alert("No nodes found matching the search term.");
            return;
        }
        
        if (matchingNodes.length === 1) {
            // If exactly one match, select that node
            this.handleNodeClick(null, matchingNodes[0]);
        } else {
            // Multiple matches, filter to show only matching nodes and their connections
            const nodeIds = new Set(matchingNodes.map(n => n.id));
            
            // Find links between matching nodes
            const relevantLinks = this.allLinks.filter(link => {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                return nodeIds.has(sourceId) || nodeIds.has(targetId);
            });
            
            // Add connected nodes to results
            relevantLinks.forEach(link => {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                nodeIds.add(sourceId);
                nodeIds.add(targetId);
            });
            
            // Update graph with filtered nodes
            const filteredNodes = this.allNodes.filter(n => nodeIds.has(n.id));
            const filteredLinks = this.allLinks.filter(link => {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                return nodeIds.has(sourceId) && nodeIds.has(targetId);
            });
            
            this.updateGraph(filteredNodes, filteredLinks);
            this.filterActive = true;
        }
    }
    
    changeLayout(layoutType) {
        // Adjust forces based on selected layout
        switch(layoutType) {
            case 'force':
                this.simulation
                    .force("link", d3.forceLink().id(d => d.id).distance(this.options.linkDistance))
                    .force("charge", d3.forceManyBody().strength(this.options.chargeStrength))
                    .force("center", d3.forceCenter(this.width / 2, this.height / 2))
                    .force("collision", d3.forceCollide().radius(this.options.nodeRadius * 1.5))
                    .force("x", null)
                    .force("y", null);
                break;
                
            case 'radial':
                this.simulation
                    .force("link", d3.forceLink().id(d => d.id).distance(this.options.linkDistance))
                    .force("charge", d3.forceManyBody().strength(this.options.chargeStrength * 0.7))
                    .force("r", d3.forceRadial(this.width / 3, this.width / 2, this.height / 2))
                    .force("center", d3.forceCenter(this.width / 2, this.height / 2))
                    .force("collision", d3.forceCollide().radius(this.options.nodeRadius * 1.2));
                break;
                
            case 'grid':
                // Create a grid arrangement
                const nodeCount = this.nodes.length;
                const cols = Math.ceil(Math.sqrt(nodeCount));
                const gridSize = Math.min(this.width, this.height) * 0.8;
                const gridStep = gridSize / cols;
                
                this.nodes.forEach((node, i) => {
                    // Assign initial positions in a grid
                    const col = i % cols;
                    const row = Math.floor(i / cols);
                    node.x = (this.width / 2 - gridSize / 2) + col * gridStep + gridStep / 2;
                    node.y = (this.height / 2 - gridSize / 2) + row * gridStep + gridStep / 2;
                });
                
                this.simulation
                    .force("link", d3.forceLink().id(d => d.id).distance(gridStep * 0.8))
                    .force("charge", d3.forceManyBody().strength(-30))
                    .force("x", d3.forceX().strength(0.1).x(d => d.x))
                    .force("y", d3.forceY().strength(0.1).y(d => d.y))
                    .force("collision", d3.forceCollide().radius(this.options.nodeRadius * 1.2));
                break;
        }
        
        // Restart simulation
        this.simulation.alpha(1).restart();
    }
    
    filterByCategory(category) {
        if (!category || category === 'all') {
            // Show all nodes
            this.resetFilters();
            return;
        }
        
        // Filter nodes by category
        const filteredNodes = this.allNodes.filter(node => node.category === category);
        
        // Find links between these nodes
        const nodeIds = new Set(filteredNodes.map(n => n.id));
        const filteredLinks = this.allLinks.filter(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return nodeIds.has(sourceId) && nodeIds.has(targetId);
        });
        
        this.updateGraph(filteredNodes, filteredLinks);
        this.filterActive = true;
    }
    
    filterByRelationship(relationshipType) {
        if (!relationshipType || relationshipType === 'all') {
            // Show all links
            this.resetFilters();
            return;
        }
        
        // Filter links by relationship type
        const filteredLinks = this.allLinks.filter(link => link.type === relationshipType);
        
        // Find nodes connected by these links
        const nodeIds = new Set();
        filteredLinks.forEach(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            nodeIds.add(sourceId);
            nodeIds.add(targetId);
        });
        
        const filteredNodes = this.allNodes.filter(node => nodeIds.has(node.id));
        
        this.updateGraph(filteredNodes, filteredLinks);
        this.filterActive = true;
    }
    
    updateLegend() {
        // Get unique relationship types
        const relationshipTypes = [...new Set(this.links.map(l => l.type))];
        
        // Create legend
        this.legendContainer.html('');
        
        // Add category legend
        const categories = [...new Set(this.nodes.map(n => n.category))];
        
        // Relationship types legend
        if (relationshipTypes.length > 0) {
            this.legendContainer.append("div")
                .text("Relationship Types:")
                .attr("class", "legend-title mb-2");
            
            const relItems = this.legendContainer.selectAll(".legend-relationship")
                .data(relationshipTypes)
                .enter()
                .append("div")
                .attr("class", "legend-item mb-1")
                .style("cursor", "pointer")
                .on("click", (event, d) => {
                    // Dispatch filter event
                    document.dispatchEvent(new CustomEvent('graphRelationshipFilterEvent', {
                        detail: { relationship: d }
                    }));
                });
            
            relItems.append("span")
                .attr("class", "legend-color")
                .style("background", "#999");
            
            relItems.append("span")
                .text(d => d);
        }
        
        // Node categories legend
        if (categories.length > 0) {
            this.legendContainer.append("hr");
            
            this.legendContainer.append("div")
                .text("Node Categories:")
                .attr("class", "legend-title mt-3 mb-2");
            
            const catItems = this.legendContainer.selectAll(".legend-category")
                .data(categories)
                .enter()
                .append("div")
                .attr("class", "legend-item mb-1")
                .style("cursor", "pointer")
                .on("click", (event, d) => {
                    // Dispatch filter event
                    document.dispatchEvent(new CustomEvent('graphCategoryFilterEvent', {
                        detail: { category: d }
                    }));
                });
            
            catItems.append("span")
                .attr("class", "legend-color")
                .style("background", d => this.getCategoryColor(d));
            
            catItems.append("span")
                .text(d => d);
        }
    }
    
    updateCategoryFilter(categories) {
        // Update category filter dropdown
        const filterSelect = document.getElementById('category-filter');
        if (filterSelect) {
            // Clear existing options
            filterSelect.innerHTML = '<option value="all">All Categories</option>';
            
            // Add option for each category
            categories.forEach(category => {
                const option = document.createElement('option');
                option.value = category;
                option.textContent = category;
                filterSelect.appendChild(option);
            });
        }
    }
    
    updateRelationshipFilter(relationships) {
        // Update relationship filter dropdown
        const filterSelect = document.getElementById('relationship-filter');
        if (filterSelect) {
            // Clear existing options
            filterSelect.innerHTML = '<option value="all">All Relationships</option>';
            
            // Add option for each relationship type
            relationships.forEach(rel => {
                const option = document.createElement('option');
                option.value = rel;
                option.textContent = rel;
                filterSelect.appendChild(option);
            });
        }
    }
    
    displayError(message) {
        this.container.innerHTML = `
            <div class="alert alert-danger" role="alert">
                ${message}
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
}
