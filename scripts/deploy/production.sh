#!/usr/bin/env bash
# AEGIS LLM Chain - Production Deployment Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_ROOT/infrastructure/docker"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   AEGIS LLM Chain Deployment Script   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Function to print colored messages
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

# Check if Docker is installed
check_docker() {
    print_info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if .env file exists
check_env_file() {
    print_info "Checking environment configuration..."
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        print_warning ".env file not found"
        print_info "Creating .env from .env.example..."
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        print_warning "Please edit .env file with your actual configuration"
        print_warning "Especially set your HF_TOKEN!"
        exit 1
    fi
    print_success "Environment file exists"
}

# Check if HF_TOKEN is set
check_hf_token() {
    print_info "Checking HuggingFace token..."
    
    # Source .env file
    if [ -f "$PROJECT_ROOT/.env" ]; then
        export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
    fi
    
    if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
        print_error "HF_TOKEN not set or using default value"
        print_warning "Please set your HuggingFace token in .env file"
        print_info "Get your token from: https://huggingface.co/settings/tokens"
        exit 1
    fi
    
    print_success "HF_TOKEN is configured"
}

# Pull latest code
pull_latest() {
    print_info "Checking for updates..."
    if [ -d "$PROJECT_ROOT/.git" ]; then
        cd "$PROJECT_ROOT"
        git fetch
        LOCAL=$(git rev-parse @)
        REMOTE=$(git rev-parse @{u})
        
        if [ "$LOCAL" != "$REMOTE" ]; then
            print_warning "Updates available"
            read -p "Pull latest changes? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git pull
                print_success "Code updated"
            fi
        else
            print_success "Code is up to date"
        fi
    fi
}

# Build Docker images
build_images() {
    print_info "Building Docker images..."
    cd "$DOCKER_DIR"
    
    if docker compose build --parallel; then
        print_success "Docker images built successfully"
    else
        print_error "Failed to build Docker images"
        exit 1
    fi
}

# Start services
start_services() {
    print_info "Starting services..."
    cd "$DOCKER_DIR"
    
    if docker compose up -d; then
        print_success "Services started successfully"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Wait for services to be healthy
wait_for_health() {
    print_info "Waiting for services to be healthy..."
    
    MAX_ATTEMPTS=30
    ATTEMPT=0
    
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "Backend is healthy"
            return 0
        fi
        
        ATTEMPT=$((ATTEMPT + 1))
        echo -n "."
        sleep 2
    done
    
    echo ""
    print_error "Backend health check failed after $MAX_ATTEMPTS attempts"
    print_info "Check logs with: docker-compose logs backend"
    return 1
}

# Show service status
show_status() {
    print_info "Service Status:"
    cd "$DOCKER_DIR"
    docker compose ps
    echo ""
}

# Show URLs
show_urls() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║      Deployment Successful! 🚀         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Available Services:${NC}"
    echo -e "  ${GREEN}•${NC} Backend API:       http://localhost:8000"
    echo -e "  ${GREEN}•${NC} API Documentation: http://localhost:8000/docs"
    echo -e "  ${GREEN}•${NC} Frontend:          http://localhost:3000"
    echo -e "  ${GREEN}•${NC} PostgreSQL:        localhost:5432"
    echo -e "  ${GREEN}•${NC} Redis:             localhost:6379"
    echo -e "  ${GREEN}•${NC} pgAdmin:           http://localhost:5050"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo -e "  ${GREEN}•${NC} View logs:         docker-compose logs -f"
    echo -e "  ${GREEN}•${NC} Stop services:     docker-compose down"
    echo -e "  ${GREEN}•${NC} Restart backend:   docker-compose restart backend"
    echo ""
}

# Main deployment flow
main() {
    echo "Starting deployment..."
    echo ""
    
    check_docker
    check_env_file
    check_hf_token
    pull_latest
    
    echo ""
    read -p "Continue with deployment? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        print_warning "Deployment cancelled"
        exit 0
    fi
    
    build_images
    start_services
    
    if wait_for_health; then
        show_status
        show_urls
    else
        print_error "Deployment completed but health check failed"
        print_info "Check logs for more information"
        exit 1
    fi
}

# Run main function
main "$@"
