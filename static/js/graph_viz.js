// Knowledge Graph Visualization using D3.js
class KnowledgeGraphVisualizer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.width = this.container.offsetWidth;
        this.height = 700;
        this.options = Object.assign({
            nodeRadius: 14,
            linkDistance: 180,
            chargeStrength: -500,
            colors: d3.scaleOrdinal(d3.schemeCategory10),
            maxZoom: 4,
            minZoom: 0.2,
            highlightRadius: 28,
            defaultRadius: 14,
            nodeOpacity: 0.9,
            labelFontSize: 12,
            clusterForce: 0.1,
            enableClustering: true,
            enablePhysics: true,
            stabilizationIterations: 50,
            // Progressive disclosure settings
            initialNodeLimit: 20,           // Initial number of nodes to show
            initialNodeSelectionStrategy: 'degree', // 'degree', 'random', 'category'
            expansionIncrement: 10,         // How many nodes to add when expanding
            expansionStrategy: 'neighbors', // How to choose nodes to expand
            highDegreeThreshold: 5,         // Threshold for "high degree" nodes
            animateDuration: 750,           // Animation duration in ms 
            progressiveDisclosureEnabled: false // Whether progressive disclosure is enabled
        }, options);
        
        // Visualization elements
        this.svg = null;
        this.zoomHandler = null;
        this.simulation = null;
        this.nodeElements = null;
        this.linkElements = null;
        this.textElements = null;
        this.linkLabels = null;
        this.markerElements = null;
        this.allNodes = []; // Store all nodes for filtering
        this.allLinks = []; // Store all links for filtering
        this.nodes = [];    // Currently visible nodes
        this.links = [];    // Currently visible links
        this.selectedNode = null; // Track currently selected node
        this.selectedLink = null; // Track currently selected relationship
        this.nodeCategories = new Set(); // Store node categories
        this.relationshipTypes = new Set(); // Store relationship types for color coding
        this.relationshipColorScale = d3.scaleOrdinal(d3.schemeSet3); // Color scale for relationships
        this.currentLayout = 'force'; // Track current layout mode
        this.currentTheme = 'dark'; // Track current theme
        this.nodeThemes = {
            dark: {
                fill: d3.scaleOrdinal(d3.schemeSet3),
                stroke: '#333',
                text: '#fff'
            },
            light: {
                fill: d3.scaleOrdinal(d3.schemePastel1),
                stroke: '#ddd',
                text: '#000'
            },
            custom: {
                fill: d3.scaleOrdinal(['#5E3E9C', '#4A90E2', '#43B88E', '#E2725B', '#E2CC5B', '#B05B85']),
                stroke: '#444',
                text: '#eee'
            }
        };
        
        // Node visibility tracking
        this.hiddenNodes = new Set();
        this.hiddenLinks = new Set();
        this.filterActive = false;
        
        // Export configuration
        this.exportInProgress = false;
        
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
        
        // Initialize the force simulation with more static parameters
        this.simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id)
                .distance(80) // Fixed distance for more predictable layout
                .strength(0.8)) // Stronger links for more stability
            .force("charge", d3.forceManyBody()
                .strength(-300) // Fixed charge for more consistent repulsion
                .distanceMax(300)) // Limited distance for more compact layout
            .force("center", d3.forceCenter(this.width / 2, this.height / 2))
            .force("collision", d3.forceCollide().radius(this.options.nodeRadius * 1.5)) // Simple collision detection
            .force("x", d3.forceX(this.width / 2).strength(0.1)) // Stronger center pull
            .force("y", d3.forceY(this.height / 2).strength(0.1)) // Stronger center pull
            .alphaDecay(0.05) // Faster decay to reach static state quicker
            .alphaTarget(0) // Target zero (static) state
            .alpha(1)
            .velocityDecay(0.5) // Higher damping for less movement
            .on("tick", () => this.ticked())
            .on("end", () => this.onSimulationEnd()); // Add handler for when simulation ends
            
        // Set a timeout to stop the simulation after a brief period anyway
        setTimeout(() => this.stopSimulation(), 2000);
        
        // Create defs for markers
        const defs = this.svg.append("defs");
        
        // Basic relationship colors - we'll dynamically add more based on actual relationships
        const markerColors = [
            {id: 'e6194b', color: '#e6194B'},  // Red
            {id: '3cb44b', color: '#3cb44b'},  // Green
            {id: '4363d8', color: '#4363d8'},  // Blue
            {id: 'f58231', color: '#f58231'},  // Orange
            {id: '911eb4', color: '#911eb4'},  // Purple
            {id: '42d4f4', color: '#42d4f4'},  // Cyan
            {id: 'f032e6', color: '#f032e6'},  // Magenta
            {id: 'bfef45', color: '#bfef45'},  // Lime
            {id: 'fabed4', color: '#fabed4'},  // Pink
        ];
        
        // Create arrow markers with different colors
        defs.selectAll('marker')
            .data(markerColors)
            .enter().append('marker')
            .attr('id', d => `marker-${d.id}`)
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', d => d.color);
            
        // Create a default marker as fallback
        defs.append('marker')
            .attr('id', 'end')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#999');
        
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
                
                // Populate relationship filter and store for coloring
                const relationshipTypes = [...new Set(this.allLinks.map(l => l.type))];
                this.relationshipTypes = new Set(relationshipTypes);
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
        
        // Create relationships first (so they appear behind nodes) with color coding
        this.linkElements = this.g.selectAll(".link")
            .data(this.links)
            .join(
                enter => enter.append("line")
                    .attr("class", d => `link link-type-${d.type.replace(/\s+/g, '-').toLowerCase()}`)
                    .attr("stroke", d => this.getRelationshipColor(d.type))
                    .attr("stroke-opacity", 0.7)
                    .attr("stroke-width", 1.8)
                    .attr("data-source", d => d.source.id || d.source)
                    .attr("data-target", d => d.target.id || d.target)
                    .attr("data-type", d => d.type)
                    .attr("marker-end", d => `url(#marker-${this.getRelationshipColorKey(d.type)})`)
                    .on("mouseover", (event, d) => this.handleLinkMouseOver(event, d))
                    .on("mouseout", () => this.handleMouseOut())
                    .on("click", (event, d) => this.handleLinkClick(event, d)),
                update => update
                    .attr("stroke", d => this.getRelationshipColor(d.type)) 
                    .attr("marker-end", d => `url(#marker-${this.getRelationshipColorKey(d.type)})`),
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
        
        // Make graph static after a brief period
        setTimeout(() => this.stopSimulation(), 2000);
    }
    
    getCategoryColor(category) {
        // Consistent colors for node categories with improved visibility
        const categoryColors = {
            'Person': '#ff9933',      // Brighter Orange
            'Movie': '#3399ff',       // Brighter Blue
            'Organization': '#33cc33', // Brighter Green
            'Genre': '#e63939',       // Brighter Red
            'Other': '#9966cc'        // Brighter Purple
        };
        return categoryColors[category] || '#9966cc';
    }
    
    // Get color for relationship types
    getRelationshipColor(relationshipType) {
        // Color palette for relationship types based on common movie relationships
        const colorMap = {
            'ACTED_IN': '#e6194B',     // Red
            'DIRECTED': '#3cb44b',     // Green
            'PRODUCED': '#4363d8',     // Blue
            'WROTE': '#f58231',        // Orange
            'BELONGS_TO': '#911eb4',   // Purple
            'RATED': '#42d4f4',        // Cyan
            'RELEASED_IN': '#f032e6',  // Magenta
            'REVIEWED': '#bfef45',     // Lime
            'FOLLOWS': '#fabed4',      // Pink
            'BASED_ON': '#469990',     // Teal
            'RELATED_TO': '#dcbeff',   // Lavender
            'APPEARS_IN': '#9A6324',   // Brown
            'SIMILAR_TO': '#fffac8',   // Beige
            'PART_OF': '#800000',      // Maroon
            'WORKED_WITH': '#aaffc3',  // Mint
        };
        
        // Clean up relationship type
        const cleanType = relationshipType.replace(/\s+/g, '_').toUpperCase();
        
        // Return color if it exists in map, otherwise get a deterministic color from the scheme
        if (colorMap[cleanType]) {
            return colorMap[cleanType];
        } else {
            // Use deterministic index for consistent coloring
            const relationshipTypes = Array.from(this.relationshipTypes);
            const index = relationshipTypes.indexOf(relationshipType) % d3.schemeSet3.length;
            return d3.schemeSet3[index];
        }
    }
    
    // Get relationship color key (for arrow markers)
    getRelationshipColorKey(relationshipType) {
        // Convert relationship type to a color key for arrow markers
        const color = this.getRelationshipColor(relationshipType);
        // Remove # and convert to lowercase
        return color.replace('#', '').toLowerCase();
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
                // Keep nodes fixed in place
                d.fx = d.x;
                d.fy = d.y;
                
                // Stop the simulation after a brief moment to let nodes settle
                setTimeout(() => this.stopSimulation(), 500);
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
    
    handleLinkClick(event, d) {
        // Toggle relationship selection
        if (this.selectedLink === d) {
            // Deselect current relationship
            this.selectedLink = null;
            
            // Reset highlighting
            this.resetHighlighting();
            
            // Dispatch event for UI updates
            document.dispatchEvent(new CustomEvent('linkDeselected'));
        } else {
            // Select new relationship
            this.selectedLink = d;
            
            // Reset node selection if any
            this.selectedNode = null;
            
            // Get source and target nodes as objects
            const source = typeof d.source === 'object' ? d.source : 
                this.nodes.find(node => node.id === d.source);
            const target = typeof d.target === 'object' ? d.target : 
                this.nodes.find(node => node.id === d.target);
            
            // Highlight just this relationship
            this.nodeElements.attr("opacity", node => 
                (node === source || node === target) ? 1 : 0.3);
            this.linkElements.attr("opacity", link => link === d ? 1 : 0.2);
            this.textElements.attr("opacity", node => 
                (node === source || node === target) ? 1 : 0.3);
            
            // Show link label
            this.linkLabelElements.attr("opacity", link => link === d ? 1 : 0);
            
            // Dispatch event with relationship details
            document.dispatchEvent(new CustomEvent('linkSelected', {
                detail: {
                    link: d,
                    source: source,
                    target: target
                }
            }));
        }
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
    
    countNodeConnections(id) {
        // Count the number of connections for a node
        if (!id || !this.allLinks) return 0;
        
        let count = 0;
        for (const link of this.allLinks) {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            
            if (sourceId === id || targetId === id) {
                count++;
            }
        }
        return count;
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
        
        // Restart simulation briefly to apply layout
        this.simulation.alpha(1).restart();
        
        // Then stop after a short time to make it static
        setTimeout(() => this.stopSimulation(), 1500);
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
        
        // Stop simulation after resize to maintain static layout
        setTimeout(() => this.stopSimulation(), 500);
    }
    
    // Method to stop the simulation completely for a static layout
    stopSimulation() {
        // Set alpha to 0 and stop the simulation
        this.simulation.alpha(0);
        this.simulation.stop();
        
        // Fix node positions
        if (this.nodes) {
            this.nodes.forEach(node => {
                node.fx = node.x;
                node.fy = node.y;
            });
        }
        
        console.log("Graph simulation stopped - static layout applied");
    }
    
    // Handler for when simulation naturally reaches end state
    onSimulationEnd() {
        console.log("Graph simulation reached end state");
        this.stopSimulation();
    }
    
    // Progressive Disclosure Methods
    enableProgressiveDisclosure(enabled) {
        // Toggle progressive disclosure mode
        this.options.progressiveDisclosureEnabled = enabled;
        
        if (enabled) {
            // Reset the current view and show only the initial nodes
            this.initializeProgressiveView();
        } else {
            // Show the full graph again
            this.resetDisclosure();
        }
    }
    
    initializeProgressiveView() {
        if (!this.allNodes.length) return;
        
        // Store current full graph
        this.fullNodes = [...this.nodes];
        this.fullLinks = [...this.links];
        
        // Determine nodes to show initially
        let initialNodes = [];
        const strategy = this.options.initialNodeSelectionStrategy;
        
        if (strategy === 'degree') {
            // Calculate the degree (number of connections) for each node
            const nodeDegrees = new Map();
            this.allNodes.forEach(node => {
                nodeDegrees.set(node.id, 0);
            });
            
            this.allLinks.forEach(link => {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                
                nodeDegrees.set(sourceId, (nodeDegrees.get(sourceId) || 0) + 1);
                nodeDegrees.set(targetId, (nodeDegrees.get(targetId) || 0) + 1);
            });
            
            // Sort nodes by degree (descending)
            const sortedNodes = [...this.allNodes].sort((a, b) => {
                return (nodeDegrees.get(b.id) || 0) - (nodeDegrees.get(a.id) || 0);
            });
            
            // Get top N nodes by degree
            initialNodes = sortedNodes.slice(0, this.options.initialNodeLimit);
        } else if (strategy === 'random') {
            // Randomly select the initial nodes
            const shuffled = [...this.allNodes].sort(() => 0.5 - Math.random());
            initialNodes = shuffled.slice(0, this.options.initialNodeLimit);
        } else if (strategy === 'category') {
            // Get a balanced selection across categories
            const categoryCounts = {};
            const nodesPerCategory = this.options.initialNodeLimit / this.nodeCategories.size;
            
            // Group nodes by category
            const nodesByCategory = {};
            this.allNodes.forEach(node => {
                if (!nodesByCategory[node.category]) {
                    nodesByCategory[node.category] = [];
                }
                nodesByCategory[node.category].push(node);
            });
            
            // Take an equal number from each category
            for (const category of this.nodeCategories) {
                if (nodesByCategory[category]) {
                    // Sort by degree or select randomly within category
                    const categoryNodes = nodesByCategory[category].slice(0, Math.ceil(nodesPerCategory));
                    initialNodes = initialNodes.concat(categoryNodes);
                }
            }
            
            // If we have too few, supplement with high-degree nodes
            if (initialNodes.length < this.options.initialNodeLimit) {
                const remainingCount = this.options.initialNodeLimit - initialNodes.length;
                const initialNodeIds = new Set(initialNodes.map(n => n.id));
                
                // Calculate degrees for remaining nodes
                const nodeDegrees = {};
                this.allLinks.forEach(link => {
                    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                    
                    nodeDegrees[sourceId] = (nodeDegrees[sourceId] || 0) + 1;
                    nodeDegrees[targetId] = (nodeDegrees[targetId] || 0) + 1;
                });
                
                // Sort remaining nodes by degree
                const remainingNodes = this.allNodes.filter(n => !initialNodeIds.has(n.id))
                    .sort((a, b) => (nodeDegrees[b.id] || 0) - (nodeDegrees[a.id] || 0));
                
                // Add the highest-degree remaining nodes
                initialNodes = initialNodes.concat(remainingNodes.slice(0, remainingCount));
            }
        }
        
        // Get initial links (only those between initial nodes)
        const initialNodeIds = new Set(initialNodes.map(n => n.id));
        const initialLinks = this.allLinks.filter(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return initialNodeIds.has(sourceId) && initialNodeIds.has(targetId);
        });
        
        // Update visible nodes and links
        this.updateGraph(initialNodes, initialLinks);
        
        // Set current expansion state
        this.expandedNodeIds = new Set(initialNodeIds);
        
        // Restart simulation
        if (this.simulation) {
            this.simulation.nodes(this.nodes);
            this.simulation.force("link").links(this.links);
            this.simulation.alpha(0.3).restart();
        }
    }
    
    resetDisclosure() {
        // Restore full graph
        this.updateGraph(this.allNodes, this.allLinks);
        
        // Restart simulation
        if (this.simulation) {
            this.simulation.nodes(this.nodes);
            this.simulation.force("link").links(this.links);
            this.simulation.alpha(0.3).restart();
        }
        
        // Reset expansion state
        this.expandedNodeIds = null;
        
        // Reset disclosure setting
        this.options.progressiveDisclosureEnabled = false;
    }
    
    expandOneLevel() {
        if (!this.options.progressiveDisclosureEnabled || !this.expandedNodeIds) {
            // Initialize progressive view if not already done
            this.options.progressiveDisclosureEnabled = true;
            this.initializeProgressiveView();
            return;
        }
        
        // Find nodes connected to already-expanded nodes
        const candidateNodes = [];
        
        this.allLinks.forEach(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            
            // If one end is in the expanded set but the other isn't, add the unexpanded node
            if (this.expandedNodeIds.has(sourceId) && !this.expandedNodeIds.has(targetId)) {
                const node = this.allNodes.find(n => n.id === targetId);
                if (node) candidateNodes.push(node);
            } else if (this.expandedNodeIds.has(targetId) && !this.expandedNodeIds.has(sourceId)) {
                const node = this.allNodes.find(n => n.id === sourceId);
                if (node) candidateNodes.push(node);
            }
        });
        
        // Remove duplicates
        const uniqueCandidates = candidateNodes.filter((node, index, self) => 
            index === self.findIndex(n => n.id === node.id));
        
        // Sort by degree or other criteria
        uniqueCandidates.sort((a, b) => {
            const connectionsA = this.countNodeConnections(a.id);
            const connectionsB = this.countNodeConnections(b.id);
            return connectionsB - connectionsA;
        });
        
        // Select nodes to add
        const nodesToAdd = uniqueCandidates.slice(0, this.options.expansionIncrement);
        
        // Add selected nodes to the expanded set
        nodesToAdd.forEach(node => this.expandedNodeIds.add(node.id));
        
        // Create the new combined node set
        const newNodes = [...this.nodes, ...nodesToAdd];
        
        // Get links for the expanded node set
        const newLinks = this.allLinks.filter(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return this.expandedNodeIds.has(sourceId) && this.expandedNodeIds.has(targetId);
        });
        
        // Update the graph
        this.updateGraph(newNodes, newLinks);
        
        // Restart simulation with longer alpha decay to allow for smoother animation
        if (this.simulation) {
            this.simulation.nodes(this.nodes);
            this.simulation.force("link").links(this.links);
            this.simulation.alpha(0.5).alphaDecay(0.02).restart();
        }
        
        // Update badge to show expansion progress
        document.getElementById('graph-filter-badge').style.display = 'inline-block';
        document.getElementById('graph-filter-badge').textContent = 
            `Progressive View (${this.nodes.length}/${this.allNodes.length} nodes)`;
    }
    
    focusOnKeyNodes() {
        // Identify high-degree nodes (key nodes in the graph)
        const nodeDegrees = {};
        
        // Calculate the degree of each node
        this.allLinks.forEach(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            
            nodeDegrees[sourceId] = (nodeDegrees[sourceId] || 0) + 1;
            nodeDegrees[targetId] = (nodeDegrees[targetId] || 0) + 1;
        });
        
        // Find nodes with degree above threshold
        const keyNodeIds = new Set();
        this.allNodes.forEach(node => {
            if ((nodeDegrees[node.id] || 0) >= this.options.highDegreeThreshold) {
                keyNodeIds.add(node.id);
            }
        });
        
        // Select the key nodes
        const keyNodes = this.allNodes.filter(node => keyNodeIds.has(node.id));
        
        // Get links between key nodes
        const keyLinks = this.allLinks.filter(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return keyNodeIds.has(sourceId) && keyNodeIds.has(targetId);
        });
        
        // Update the graph
        this.updateGraph(keyNodes, keyLinks);
        
        // Restart simulation
        if (this.simulation) {
            this.simulation.nodes(this.nodes);
            this.simulation.force("link").links(this.links);
            this.simulation.alpha(0.3).restart();
        }
        
        // Update UI to reflect filtering
        document.getElementById('graph-filter-badge').style.display = 'inline-block';
        document.getElementById('graph-filter-badge').textContent = 
            `Key Nodes (${keyNodes.length} nodes with degree ≥ ${this.options.highDegreeThreshold})`;
    }
    
    // Theme management
    applyTheme(themeName) {
        // Apply the selected theme to the graph visualization
        this.currentTheme = themeName;
        const theme = this.nodeThemes[themeName] || this.nodeThemes.dark;
        
        // Update node colors
        this.svg.selectAll(".node")
            .transition()
            .duration(500)
            .attr("fill", d => theme.fill(d.category))
            .attr("stroke", theme.stroke);
        
        // Update text colors
        this.svg.selectAll(".node-label")
            .transition()
            .duration(500)
            .attr("fill", theme.text);
        
        // Update background if needed
        if (themeName === 'light') {
            this.svg.transition().duration(500).style("background-color", "#f8f9fa");
        } else {
            this.svg.transition().duration(500).style("background-color", "#212529");
        }
        
        // Dispatch theme change event for potential UI updates
        document.dispatchEvent(new CustomEvent('graphThemeChanged', {
            detail: { theme: themeName }
        }));
    }
    
    updateCustomPalette(colors) {
        // Update the custom theme with new colors
        this.nodeThemes.custom.fill = d3.scaleOrdinal(colors);
        
        // If current theme is custom, apply changes immediately
        if (this.currentTheme === 'custom') {
            this.applyTheme('custom');
        }
    }
    
    // Export graph functionality
    exportGraph(format = 'png') {
        if (this.exportInProgress) return;
        this.exportInProgress = true;
        
        // Show loading spinner or indicator
        const statusEl = document.getElementById('export-status');
        const spinnerEl = document.getElementById('export-spinner');
        
        if (statusEl) statusEl.textContent = 'Preparing export...';
        if (spinnerEl) spinnerEl.classList.remove('d-none');
        
        try {
            // Clone the SVG for export to avoid modifying the original
            const originalSvg = document.querySelector(`#${this.containerId} svg`);
            if (!originalSvg) {
                throw new Error("SVG element not found");
            }
            
            const svgClone = originalSvg.cloneNode(true);
            
            // Add dark background for better visibility in export
            const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            rect.setAttribute("width", "100%");
            rect.setAttribute("height", "100%");
            rect.setAttribute("fill", "#121212");
            svgClone.insertBefore(rect, svgClone.firstChild);
            
            // Set fixed dimensions for export
            const width = this.width || 1200;  // Default if not set
            const height = this.height || 800;  // Default if not set
            svgClone.setAttribute("width", width);
            svgClone.setAttribute("height", height);
            svgClone.setAttribute("viewBox", `0 0 ${width} ${height}`);
            
            // Convert to data URL
            const svgData = new XMLSerializer().serializeToString(svgClone);
            const svgBlob = new Blob([svgData], {type: "image/svg+xml;charset=utf-8"});
            
            // Current timestamp for filename
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
            
            if (format === 'svg') {
                // Direct SVG download
                this.downloadBlob(svgBlob, `knowledge_graph_${timestamp}.svg`);
                this.exportInProgress = false;
                if (statusEl) statusEl.textContent = 'Export complete';
                if (spinnerEl) spinnerEl.classList.add('d-none');
                
                // Reset status after delay
                setTimeout(() => {
                    if (statusEl) statusEl.textContent = '';
                }, 3000);
            } else {
                // Convert to PNG using data URL approach first, fallback to canvas if needed
                const svgUrl = URL.createObjectURL(svgBlob);
                
                // Create image from SVG
                const img = new Image();
                
                img.onload = () => {
                    try {
                        // Create canvas for conversion
                        const canvas = document.createElement("canvas");
                        // Use a scale factor for higher quality
                        const scale = window.devicePixelRatio || 2;
                        canvas.width = width * scale;
                        canvas.height = height * scale;
                        const ctx = canvas.getContext("2d");
                        
                        if (!ctx) {
                            throw new Error("Could not get canvas context");
                        }
                        
                        // Apply scaling for high resolution
                        ctx.scale(scale, scale);
                        
                        // Fill background
                        ctx.fillStyle = "#121212";
                        ctx.fillRect(0, 0, width, height);
                        
                        // Draw image
                        ctx.drawImage(img, 0, 0, width, height);
                        
                        // Convert to blob and download
                        canvas.toBlob((blob) => {
                            if (blob) {
                                this.downloadBlob(blob, `knowledge_graph_${timestamp}.png`);
                                if (statusEl) statusEl.textContent = 'Export complete';
                            } else {
                                console.error("Failed to create blob from canvas");
                                if (statusEl) statusEl.textContent = 'Export failed - trying alternative method...';
                                
                                // Fallback method: Use data URL
                                const dataUrl = canvas.toDataURL('image/png');
                                const dl = document.createElement('a');
                                dl.href = dataUrl;
                                dl.download = `knowledge_graph_${timestamp}.png`;
                                document.body.appendChild(dl);
                                dl.click();
                                document.body.removeChild(dl);
                                
                                if (statusEl) statusEl.textContent = 'Export complete (alternative method)';
                            }
                            
                            // Clean up
                            URL.revokeObjectURL(svgUrl);
                            this.exportInProgress = false;
                            if (spinnerEl) spinnerEl.classList.add('d-none');
                            
                            // Reset status after delay
                            setTimeout(() => {
                                if (statusEl) statusEl.textContent = '';
                            }, 3000);
                        }, 'image/png', 1.0);
                    } catch (canvasError) {
                        console.error("Canvas error:", canvasError);
                        if (statusEl) statusEl.textContent = 'Export failed - trying SVG fallback...';
                        
                        // If canvas fails, use SVG download as fallback
                        this.downloadBlob(svgBlob, `knowledge_graph_${timestamp}.svg`);
                        if (statusEl) statusEl.textContent = 'Export complete (SVG fallback)';
                        
                        // Clean up
                        URL.revokeObjectURL(svgUrl);
                        this.exportInProgress = false;
                        if (spinnerEl) spinnerEl.classList.add('d-none');
                        
                        // Reset status after delay
                        setTimeout(() => {
                            if (statusEl) statusEl.textContent = '';
                        }, 3000);
                    }
                };
                
                img.onerror = (e) => {
                    console.error("Failed to load SVG as image:", e);
                    if (statusEl) statusEl.textContent = 'Export failed - trying direct SVG download...';
                    
                    // If image loading fails, fallback to SVG download
                    this.downloadBlob(svgBlob, `knowledge_graph_${timestamp}.svg`);
                    if (statusEl) statusEl.textContent = 'Export complete (SVG fallback)';
                    
                    // Clean up
                    URL.revokeObjectURL(svgUrl);
                    this.exportInProgress = false;
                    if (spinnerEl) spinnerEl.classList.add('d-none');
                    
                    // Reset status after delay
                    setTimeout(() => {
                        if (statusEl) statusEl.textContent = '';
                    }, 3000);
                };
                
                // Set image source to SVG blob URL
                img.src = svgUrl;
            }
        } catch (error) {
            console.error("Error during export:", error);
            if (statusEl) statusEl.textContent = 'Export failed';
            if (spinnerEl) spinnerEl.classList.add('d-none');
            this.exportInProgress = false;
            
            // Reset status after delay
            setTimeout(() => {
                if (statusEl) statusEl.textContent = '';
            }, 3000);
        }
    }
    
    // Helper method to download a blob with browser compatibility
    downloadBlob(blob, fileName) {
        try {
            // Method 1: Using URL.createObjectURL
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = fileName;
            link.style.display = "none";
            document.body.appendChild(link);
            
            // Trigger download with a slight delay to ensure proper attachment in all browsers
            setTimeout(() => {
                try {
                    link.click();
                    // Clean up
                    setTimeout(() => {
                        document.body.removeChild(link);
                        URL.revokeObjectURL(url);
                    }, 100);
                } catch (e) {
                    console.error("Error in click event:", e);
                    this.fallbackDownload(blob, fileName);
                }
            }, 50);
        } catch (error) {
            console.error("Primary download method failed:", error);
            this.fallbackDownload(blob, fileName);
        }
    }
    
    // Fallback download method using different techniques
    fallbackDownload(blob, fileName) {
        try {
            // Method 2: Using window.navigator.msSaveBlob (for IE/Edge)
            if (window.navigator && window.navigator.msSaveBlob) {
                window.navigator.msSaveBlob(blob, fileName);
                return;
            }
            
            // Method 3: Create data URL and trigger download
            const reader = new FileReader();
            reader.onload = function() {
                const dataUrl = reader.result;
                const link = document.createElement("a");
                link.href = dataUrl;
                link.download = fileName;
                link.style.display = "none";
                document.body.appendChild(link);
                link.click();
                setTimeout(() => {
                    document.body.removeChild(link);
                }, 100);
            };
            reader.readAsDataURL(blob);
        } catch (fallbackError) {
            console.error("Fallback download method failed:", fallbackError);
            // Show error message to user
            const statusEl = document.getElementById('export-status');
            if (statusEl) {
                statusEl.textContent = 'Browser download failed. Try a different browser.';
                setTimeout(() => {
                    statusEl.textContent = '';
                }, 5000);
            }
        }
    }
}
