#!/bin/bash

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Set up PATH for ~/.local/bin
python -c "
import os
bashrc_line = 'export PATH=\"\$HOME/.local/bin:\$PATH\"'
bashrc_path = os.path.expanduser('~/.bashrc')
if os.path.exists(bashrc_path):
    with open(bashrc_path, 'r') as f:
        content = f.read()
    if bashrc_line not in content:
        with open(bashrc_path, 'a') as f:
            f.write(f'\n{bashrc_line}\n')
else:
    with open(bashrc_path, 'a') as f:
        f.write(f'\n{bashrc_line}\n')
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':' + os.environ.get('PATH', '')
"

log_info "Detecting current zone..."
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | cut -d/ -f4)
log_info "Current zone: $ZONE"

if ! command -v gcloud &>/dev/null; then
    log_error "gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 >/dev/null; then
    log_error "No active gcloud authentication found. Please run 'gcloud auth login'"
    exit 1
fi

log_info "Searching for available TPUs in zone $ZONE..."
READY_TPUS=($(gcloud compute tpus tpu-vm list --zone="$ZONE" --filter="state:READY" --format="value(name)" 2>/dev/null))

case ${#READY_TPUS[@]} in
0)
    log_warning "No READY TPUs found in zone $ZONE"

    echo ""
    log_info "All TPUs in zone $ZONE:"
    gcloud compute tpus tpu-vm list --zone="$ZONE" --format="table(name,state,health,acceleratorType)" 2>/dev/null || {
        log_error "Failed to list TPUs. Check your permissions."
        exit 1
    }

    echo ""
    read -p "Enter your TPU name: " TPU_NAME < /dev/tty

    if [ -z "$TPU_NAME" ]; then
        log_error "TPU name cannot be empty"
        exit 1
    fi
    ;;
1)
    TPU_NAME="${READY_TPUS[0]}"
    log_success "Found one READY TPU: $TPU_NAME - using it automatically"
    ;;
*)
    log_info "Multiple READY TPUs found:"
    echo ""
    gcloud compute tpus tpu-vm list --zone="$ZONE" --filter="state:READY" --format="table(name,acceleratorType,health)"
    echo ""

    echo "Available READY TPUs:"
    for i in "${!READY_TPUS[@]}"; do
        echo "$((i + 1)). ${READY_TPUS[i]}"
    done
    echo ""

    while true; do
        read -p "Enter TPU name or number (1-${#READY_TPUS[@]}): " input < /dev/tty

        if [[ "$input" =~ ^[0-9]+$ ]] && [ "$input" -ge 1 ] && [ "$input" -le ${#READY_TPUS[@]} ]; then
            TPU_NAME="${READY_TPUS[$((input - 1))]}"
            break
        elif [[ " ${READY_TPUS[@]} " =~ " ${input} " ]]; then
            TPU_NAME="$input"
            break
        else
            log_error "Invalid selection. Please enter a number (1-${#READY_TPUS[@]}) or valid TPU name."
        fi
    done
    ;;
esac

log_success "Selected TPU: $TPU_NAME"
log_info "Installing eopod..."
if ! pip install eopod --quiet -U; then
    log_error "Failed to install eopod"
    exit 1
fi

EOPOD_PATH="$HOME/.local/bin/eopod"
if [ ! -f "$EOPOD_PATH" ]; then
    log_error "eopod not found at $EOPOD_PATH after installation"
    exit 1
fi
log_info "Configuring eopod with TPU: $TPU_NAME"
if ! "$EOPOD_PATH" configure --tpu-name "$TPU_NAME"; then
    log_error "Failed to configure eopod with TPU"
    exit 1
fi

log_success "eopod configured successfully"
log_warning "IMPORTANT: Press Enter during first execution to accept terms (terms may not be displayed)"
echo ""

read -p "Press Enter to continue with package installations..." < /dev/tty
 
VENV_PATH="$HOME/easy-venv"
log_info "Setting up virtual environment with uv at $VENV_PATH..."
if ! command -v uv &>/dev/null; then
    log_info "Installing uv..."
    if ! pip install uv --quiet -U; then
        log_error "Failed to install uv"
        exit 1
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    log_info "Creating virtual environment..."
    if ! uv venv "$VENV_PATH"; then
        log_error "Failed to create virtual environment"
        exit 1
    fi
fi

# Add virtual environment activation to ~/.bashrc
VENV_ACTIVATE="source $VENV_PATH/bin/activate"
BASHRC_PATH="$HOME/.bashrc"
if [ -f "$BASHRC_PATH" ]; then
    if ! grep -Fx "$VENV_ACTIVATE" "$BASHRC_PATH" > /dev/null; then
        log_info "Adding virtual environment activation to $BASHRC_PATH..."
        echo "$VENV_ACTIVATE" >> "$BASHRC_PATH"
    else
        log_info "Virtual environment activation already in $BASHRC_PATH"
    fi
else
    log_info "Creating $BASHRC_PATH and adding virtual environment activation..."
    echo "$VENV_ACTIVATE" > "$BASHRC_PATH"
fi

# Activate the virtual environment for the current session
source "$VENV_PATH/bin/activate"

install_package() {
    local package="$1"
    local extra_args="$2"
    log_info "Installing $package in virtual environment..." 
    if [[ "$package" == *git+* ]]; then 
        local git_url="${package#*@}"
        local pkg_name_with_extras="${package%% @*}"
        if ! uv pip install "${pkg_name_with_extras}@${git_url}" $extra_args --quiet; then
            log_error "Failed to install $package"
            return 1
        fi
    else
        if ! uv pip install "$package" $extra_args --quiet; then
            log_error "Failed to install $package"
            return 1
        fi
    fi
    log_success "Successfully installed $package"
}

echo ""
log_info "Starting package installations in virtual environment..."

log_info "Uninstalling existing easydel..."
uv pip uninstall easydel -y --quiet 2>/dev/null || true
install_package "easydel[tpu,torch] @ git+https://github.com/erfanzar/easydel.git" || exit 1

log_info "Configuring Ray..."
if ! "$EOPOD_PATH" auto-config-ray --self-job; then
    log_error "Failed to configure Ray"
    exit 1
fi

log_success "Ray configured successfully"

echo ""
log_success "🎉 TPU setup completed successfully!"
log_info "TPU Name: $TPU_NAME"
log_info "Zone: $ZONE"
echo ""
log_info "Final TPU status:"
gcloud compute tpus tpu-vm list --zone="$ZONE" --filter="name:$TPU_NAME" --format="table(name,state,health,acceleratorType)"
log_info "Virtual Environment: $VENV_PATH"
log_info "All packages installed and configured in virtual environment"