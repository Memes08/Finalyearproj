// Knowledge Graph Visualization using D3.js
class KnowledgeGraphVisualizer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.width = this.container.offsetWidth;
        this.height = 600;
        this.options = Object.assign({
            nodeRadius: 12,
            linkDistance: 150,
            chargeStrength: -300,
            colors: d3.scaleOrdinal(d3.schemeCategory10)
        }, options);
        
        this.svg = null;
        this.simulation = null;
        this.nodeElements = null;
        this.linkElements = null;
        this.textElements = null;
        
        this.nodes = [];
        this.links = [];
        
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
                this.updateGraph(data.nodes, data.relationships);
            })
            .catch(error => {
                console.error('Error loading graph data:', error);
                this.displayError('Failed to load graph data. Please try again later.');
            });
    }
    
    updateGraph(nodes, links) {
        this.nodes = nodes;
        this.links = links.map(l => ({
            source: l.source,
            target: l.target,
            type: l.type
        }));
        
        // Create relationships first (so they appear behind nodes)
        this.linkElements = this.g.selectAll(".link")
            .data(this.links)
            .join(
                enter => enter.append("line")
                    .attr("class", "link")
                    .attr("stroke", "#999")
                    .attr("stroke-opacity", 0.6)
                    .attr("stroke-width", 1.5)
                    .attr("marker-end", "url(#end)"),
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
                    .attr("class", "node")
                    .attr("r", this.options.nodeRadius)
                    .attr("fill", (d, i) => this.options.colors(i % 10))
                    .call(this.setupDragHandlers())
                    .on("mouseover", (event, d) => this.handleNodeMouseOver(event, d))
                    .on("mouseout", () => this.handleNodeMouseOut()),
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
        
        this.tooltipDiv.html(d.label)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
    }
    
    handleNodeMouseOut() {
        // Hide tooltip
        this.tooltipDiv.transition()
            .duration(500)
            .style("opacity", 0);
    }
    
    updateLegend() {
        // Get unique relationship types
        const relationshipTypes = [...new Set(this.links.map(l => l.type))];
        
        // Create legend
        this.legendContainer.html('');
        this.legendContainer.append("div")
            .text("Relationship Types:")
            .attr("class", "legend-title");
        
        const legendItems = this.legendContainer.selectAll(".legend-item")
            .data(relationshipTypes)
            .enter()
            .append("div")
            .attr("class", "legend-item");
        
        legendItems.append("span")
            .attr("class", "legend-color")
            .style("background", "#999");
        
        legendItems.append("span")
            .text(d => d);
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
