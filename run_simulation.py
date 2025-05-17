#!/usr/bin/env python3
"""
Traffic Simulation Launcher
============================

Easy-to-use launcher for different traffic simulation environments.
Run with: python run_simulation.py [environment_type]

Available environments:
- highway: Realistic highway traffic with dedicated lanes (RECOMMENDED)
- city: Original city traffic with RL agent
- demo: Speed variation demo with enhanced visuals
- realistic: Destination-based navigation
"""

import sys
import subprocess
import os

def run_highway():
    """Run highway traffic simulation (recommended)"""
    print("🛣️  Starting Highway Traffic Simulation...")
    print("Features: Dedicated lanes, continuous flow, realistic following, journey time tracking")
    cmd = [
        "python", "visualizers/highway_visualizer.py",
        "--episodes", "1",
        "--max-steps", "300", 
        "--delay", "0.15",
        "--grid-size", "10",
        "--max-vehicles", "24",
        "--spawn-rate", "0.4"
    ]
    subprocess.run(cmd)

def run_city():
    """Run original city traffic with trained agent"""
    print("🚗 Starting City Traffic Simulation...")
    print("Features: Multi-agent RL, intersection management")
    cmd = [
        "python", "visualizers/visualize_agent.py",
        "--episodes", "1",
        "--max-steps", "50",
        "--delay", "0.2"
    ]
    subprocess.run(cmd)

def run_demo():
    """Run demo with speed variations"""
    print("🎬 Starting Demo Traffic Simulation...")
    print("Features: Speed visualization, enhanced graphics")
    cmd = [
        "python", "visualizers/demo_visualizer.py",
        "--episodes", "1",
        "--max-steps", "150",
        "--delay", "0.25",
        "--grid-size", "8",
        "--num-vehicles", "4"
    ]
    subprocess.run(cmd)

def run_realistic():
    """Run realistic traffic with destination navigation"""
    print("🎯 Starting Realistic Traffic Simulation...")
    print("Features: Destination navigation, traffic awareness")
    cmd = [
        "python", "visualizers/realistic_visualizer.py",
        "--episodes", "1",
        "--max-steps", "150",
        "--delay", "0.3",
        "--grid-size", "6",
        "--num-vehicles", "3"
    ]
    subprocess.run(cmd)

def run_smart_city():
    """Run smart city traffic with 6G communication (short episodes)"""
    print("🌆 Starting Smart City Traffic with 6G Communication...")
    print("Features: 6G V2V/V2I communication, intersection reservations, collision prevention")
    cmd = [
        "python", "visualizers/visualize_agent.py",
        "--episodes", "1",
        "--max-steps", "200", 
        "--delay", "0.2",
        "--grid-size", "10",
        "--num-vehicles", "8"
    ]
    subprocess.run(cmd)

def run_smart_city_live():
    """Run continuous smart city traffic with 6G communication (BEST visualization)"""
    print("🌆 Starting Continuous Smart City Traffic with 6G Communication...")
    print("Features: 6G V2V/V2I communication, continuous vehicle spawning, journey analytics")
    cmd = [
        "python", "visualizers/smart_city_visualizer.py",
        "--episodes", "1",
        "--max-steps", "300", 
        "--delay", "0.15",
        "--grid-size", "10",
        "--max-vehicles", "12"
    ]
    subprocess.run(cmd)

def run_smart_highway():
    """Run smart highway with 6G communication and enhanced multi-lane traffic (PERFECT!)"""
    print("🛣️ Starting Smart Highway with 6G Communication and Multi-Lane Traffic...")
    print("Features: 2-direction system, multiple vehicles per lane, clear intersections, 6G communication")
    cmd = [
        "python", "visualizers/smart_highway_visualizer.py",
        "--episodes", "1",
        "--max-steps", "300", 
        "--delay", "0.1",
        "--grid-size", "10",
        "--max-vehicles", "24",
        "--spawn-rate", "0.6"
    ]
    subprocess.run(cmd)

def show_help():
    """Show help information"""
    print("""
🚗 Traffic Simulation Suite - Launcher
=====================================

Usage: python run_simulation.py [environment]

Available environments:
  smart_highway   - Smart highway with 6G + directional lanes (PERFECT SOLUTION!)
  smart_city_live - Continuous smart city with 6G communication (BEST VISUALIZATION!)
  smart_city      - Smart city with 6G V2V/V2I communication (short episodes)
  highway         - Highway traffic with dedicated lanes (RECOMMENDED for flow)
  city            - Original city traffic with RL agent  
  demo            - Speed variation demo with enhanced visuals
  realistic       - Destination-based navigation
  
Examples:
  python run_simulation.py smart_highway
  python run_simulation.py smart_city_live
  python run_simulation.py smart_city
  python run_simulation.py highway
  python run_simulation.py city
  python run_simulation.py demo
  python run_simulation.py realistic

If no environment is specified, smart_highway will be used (perfect solution).

Environment Comparison:
┌─────────────────┬─────────┬───────────┬──────────────────────────────────┐
│ Environment     │ 6G Comm│ Realism   │ Features                         │
├─────────────────┼─────────┼───────────┼──────────────────────────────────┤
│ smart_highway   │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐   │ X/Y-only + 6G + intersections    │
│ smart_city_live │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐     │ 6G + continuous flow + analytics │
│ smart_city      │ ⭐⭐⭐⭐⭐ │ ⭐⭐      │ Full 6G system, short episodes   │
│ highway         │ ❌       │ ⭐⭐⭐⭐⭐   │ Dedicated lanes, continuous flow │
│ city            │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐     │ Multi-agent RL, intersections    │
│ realistic       │ ⭐⭐⭐⭐  │ ⭐⭐      │ Destination navigation           │
│ demo            │ ❌       │ ⭐⭐⭐     │ Speed visualization              │
└─────────────────┴─────────┴───────────┴──────────────────────────────────┘
""")

def main():
    """Main launcher function"""
    if len(sys.argv) == 1:
        # Default to smart_highway if no argument provided (perfect solution)
        environment = "smart_highway"
    elif len(sys.argv) == 2:
        environment = sys.argv[1].lower()
    else:
        show_help()
        return
    
    # Check if we're in the right directory
    if not os.path.exists("visualizers"):
        print("❌ Error: Please run this script from the simulation_city_grid directory")
        print("   Current directory:", os.getcwd())
        return
    
    # Route to appropriate function
    if environment == "smart_highway":
        run_smart_highway()
    elif environment == "smart_city_live":
        run_smart_city_live()
    elif environment == "smart_city":
        run_smart_city()
    elif environment == "highway":
        run_highway()
    elif environment == "city":
        run_city()
    elif environment == "demo":
        run_demo()
    elif environment == "realistic":
        run_realistic()
    elif environment in ["help", "-h", "--help"]:
        show_help()
    else:
        print(f"❌ Unknown environment: {environment}")
        print("Available options: smart_highway, smart_city_live, smart_city, highway, city, demo, realistic")
        print("Use 'python run_simulation.py help' for more information")

if __name__ == "__main__":
    main() 