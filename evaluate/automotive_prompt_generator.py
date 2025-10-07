import itertools
import random
import pandas as pd

class AutomotivePromptGenerator:
    def __init__(self):
        # Define core feature categories with options
        self.features = {
            "brand": ["BMW", "Mercedes", "Audi", "Porsche", "Tesla", "Lexus"],
            
            "vehicle_type": {
                "coupe": ["sports coupe", "grand touring coupe", "compact coupe"],
                "sedan": ["sport sedan", "luxury sedan", "compact sedan"],
                "suv": ["compact SUV", "mid-size SUV", "luxury SUV", "sport SUV"],
                "wagon": ["sport wagon", "luxury estate", "touring wagon"],
                "convertible": ["roadster", "cabriolet", "sport convertible"]
            },
            
            "styling_era": ["classic", "modern", "futuristic", "retro-inspired", "contemporary"],
            
            "color": {
                "base": ["Alpine White", "Jet Black", "Mineral Grey", "Tanzanite Blue", "San Marino Blue", 
                         "Sunset Orange", "Vegas Yellow", "British Racing Green", "Fire Red", "Nardo Grey"],
                "finish": ["metallic", "matte", "gloss", "pearl", "satin"]
            },
            
            "design_elements": {
                "front": ["aggressive front fascia", "distinctive kidney grille", "slim LED headlights", 
                         "carbon fiber front splitter", "large air intakes", "illuminated grille"],
                "side": ["pronounced fender flares", "strong shoulder line", "flush door handles",
                         "aerodynamic side skirts", "contrasting side mirrors", "black A-pillars"],
                "rear": ["muscular rear haunches", "distinctive LED taillights", "integrated rear spoiler",
                        "quad exhaust tips", "carbon fiber diffuser", "full-width light bar"]
            },
            
            "wheels": {
                "size": ["18-inch", "19-inch", "20-inch", "21-inch", "22-inch"],
                "design": ["multi-spoke", "five-spoke", "split five-spoke", "Y-spoke", "turbine-style"],
                "finish": ["diamond-cut", "gloss black", "machined face", "brushed aluminum", "bronze"]
            },
            
            "special_features": ["panoramic glass roof", "carbon fiber accents", "aggressive stance", 
                                "widebody kit", "M-division styling", "sport exhaust system",
                                "blacked-out trim", "laser headlights", "functional air vents"],
            
            # "lighting": ["soft even studio lighting", "dramatic side lighting", "high-key studio lighting",
            #             "low-key lighting with strong highlights", "diffused lighting minimizing harsh reflections"],
            
            # "viewpoint": ["front 3/4 view", "rear 3/4 view", "profile view", "high angle front view", 
            #              "low angle rear view", "direct front view", "direct rear view"],
                         
            # "background": ["clean white background", "dark neutral background", "industrial studio setting",
            #               "minimalist grey background", "gradient studio background"]
        }
        
    def generate_single_prompt(self, fixed_elements=None):
        """Generate a single prompt with random selections from each category"""
        if fixed_elements is None:
            fixed_elements = {}
            
        # Initialize components
        brand = fixed_elements.get('brand', random.choice(self.features["brand"]))
        
        # Select vehicle type and subtype
        if 'vehicle_type' in fixed_elements:
            vehicle_type_category = fixed_elements['vehicle_type'].split('_')[0]
            vehicle_type = fixed_elements['vehicle_type'].split('_')[1]
        else:
            vehicle_type_category = random.choice(list(self.features["vehicle_type"].keys()))
            vehicle_type = random.choice(self.features["vehicle_type"][vehicle_type_category])
        
        # Select styling era
        styling_era = fixed_elements.get('styling_era', random.choice(self.features["styling_era"]))
        
        # Select color and finish
        color_base = fixed_elements.get('color_base', random.choice(self.features["color"]["base"]))
        color_finish = fixed_elements.get('color_finish', random.choice(self.features["color"]["finish"]))
        
        # Select design elements (1-2 from each category)
        num_front_elements = random.randint(1, 2)
        num_side_elements = random.randint(0, 1)
        num_rear_elements = random.randint(0, 1)
        
        front_elements = random.sample(self.features["design_elements"]["front"], num_front_elements)
        side_elements = random.sample(self.features["design_elements"]["side"], num_side_elements)
        rear_elements = random.sample(self.features["design_elements"]["rear"], num_rear_elements)
        
        # Combine design elements
        design_elements = front_elements + side_elements + rear_elements
        
        # Select wheels
        wheel_size = fixed_elements.get('wheel_size', random.choice(self.features["wheels"]["size"]))
        wheel_design = fixed_elements.get('wheel_design', random.choice(self.features["wheels"]["design"]))
        wheel_finish = fixed_elements.get('wheel_finish', random.choice(self.features["wheels"]["finish"]))
        
        # Select special features (0-2)
        num_special_features = random.randint(0, 2)
        special_features = fixed_elements.get('special_features', 
                                             random.sample(self.features["special_features"], num_special_features))
        
        # # Select lighting, viewpoint and background
        # lighting = fixed_elements.get('lighting', random.choice(self.features["lighting"]))
        # viewpoint = fixed_elements.get('viewpoint', random.choice(self.features["viewpoint"]))
        # background = fixed_elements.get('background', random.choice(self.features["background"]))
        
        # Construct the prompt
        prompt_parts = [
            f"{color_finish} {color_base} {brand} {styling_era} {vehicle_type}",
            f"featuring {', '.join(design_elements)}" if design_elements else "",
            f"with {wheel_size} {wheel_design} wheels in {wheel_finish}" if wheel_size else "",
            f"{', '.join(special_features)}" if special_features and len(special_features) > 0 else "",
            # f"{viewpoint} on {background}",
            # f"{lighting}"
        ]
        
        # Filter out empty parts and join
        prompt = ". ".join([part for part in prompt_parts if part])
        
        # Add the 3D rendered image prefix
        prompt = "A 3D rendered image. " + prompt
        
        return prompt
    
    def generate_attribute_variation_set(self, attribute, values=None):
        """Generate a set of prompts varying only one attribute while keeping others constant"""
        # Create a baseline prompt configuration
        baseline = {
            'brand': 'BMW',
            'vehicle_type': 'coupe_sports coupe',
            'styling_era': 'modern',
            'color_base': 'Mineral Grey',
            'color_finish': 'metallic',
            'wheel_size': '20-inch',
            'wheel_design': 'multi-spoke',
            'wheel_finish': 'diamond-cut',
            'special_features': ['aggressive stance'],
            # 'viewpoint': 'front 3/4 view',
            # 'lighting': 'soft even studio lighting',
            # 'background': 'clean white background'
        }
        
        prompts = []
        
        # If no specific values are provided, use all available values for the attribute
        if values is None:
            if attribute == 'color':
                # Special handling for color which has base and finish
                for base in self.features["color"]["base"]:
                    for finish in self.features["color"]["finish"]:
                        test_config = baseline.copy()
                        test_config['color_base'] = base
                        test_config['color_finish'] = finish
                        prompts.append(self.generate_single_prompt(test_config))
            elif attribute == 'vehicle_type':
                # Special handling for vehicle type which has categories and types
                for category in self.features["vehicle_type"]:
                    for type_value in self.features["vehicle_type"][category]:
                        test_config = baseline.copy()
                        test_config['vehicle_type'] = f"{category}_{type_value}"
                        prompts.append(self.generate_single_prompt(test_config))
            elif attribute == 'wheels':
                # Special handling for wheels which has size, design and finish
                for size in self.features["wheels"]["size"]:
                    test_config = baseline.copy()
                    test_config['wheel_size'] = size
                    prompts.append(self.generate_single_prompt(test_config))
                for design in self.features["wheels"]["design"]:
                    test_config = baseline.copy()
                    test_config['wheel_design'] = design
                    prompts.append(self.generate_single_prompt(test_config))
                for finish in self.features["wheels"]["finish"]:
                    test_config = baseline.copy()
                    test_config['wheel_finish'] = finish
                    prompts.append(self.generate_single_prompt(test_config))
            elif attribute in self.features:
                # For regular attributes
                for value in self.features[attribute]:
                    test_config = baseline.copy()
                    test_config[attribute] = value
                    prompts.append(self.generate_single_prompt(test_config))
        else:
            # Use provided values
            for value in values:
                test_config = baseline.copy()
                test_config[attribute] = value
                prompts.append(self.generate_single_prompt(test_config))
        
        return prompts
    
    def generate_comprehensive_test_set(self, sample_size=5):
        """Generate a comprehensive test set with variations of each attribute"""
        all_prompts = []
        
        # Test color variations
        color_prompts = self.generate_attribute_variation_set('color')
        all_prompts.extend(random.sample(color_prompts, min(sample_size, len(color_prompts))))
        
        # Test vehicle type variations
        vehicle_prompts = self.generate_attribute_variation_set('vehicle_type')
        all_prompts.extend(random.sample(vehicle_prompts, min(sample_size, len(vehicle_prompts))))
        
        # Test brand variations
        brand_prompts = self.generate_attribute_variation_set('brand')
        all_prompts.extend(brand_prompts)  # Include all brands
        
        # Test wheel variations
        wheel_prompts = self.generate_attribute_variation_set('wheels')
        all_prompts.extend(random.sample(wheel_prompts, min(sample_size, len(wheel_prompts))))
        
        # # Test viewpoint variations
        # viewpoint_prompts = self.generate_attribute_variation_set('viewpoint')
        # all_prompts.extend(viewpoint_prompts)  # Include all viewpoints
        
        # Test special features
        for feature in self.features["special_features"]:
            test_config = {
                'brand': 'BMW',
                'vehicle_type': 'coupe_sports coupe',
                'styling_era': 'modern',
                'color_base': 'Mineral Grey',
                'color_finish': 'metallic',
                'special_features': [feature],
                # 'viewpoint': 'front 3/4 view'
            }
            all_prompts.append(self.generate_single_prompt(test_config))
            
        # Generate some completely random prompts
        for _ in range(sample_size):
            all_prompts.append(self.generate_single_prompt())
            
        return all_prompts
        
    def export_to_csv(self, prompts, filename="automotive_test_prompts.csv"):
        """Export prompts to CSV with metadata for analysis"""
        # Create simple DataFrame with just prompts for now
        df = pd.DataFrame({"prompt": prompts})
        df.to_csv(filename, index=False)
        print(f"Exported {len(prompts)} prompts to {filename}")

    def load_from_csv(self, filename="automotive_test_prompts.csv"):
        """Load prompts from a CSV file"""
        df = pd.read_csv(filename)
        return df["prompt"].tolist()


# Example usage:
if __name__ == "__main__":
    generator = AutomotivePromptGenerator()
    
    # Generate comprehensive test set
    test_prompts = generator.generate_comprehensive_test_set(sample_size=5)
    
    # Add specific attribute tests for color
    color_test = generator.generate_attribute_variation_set('color_base', 
                                                          ["Alpine White", "Jet Black", "Fire Red", 
                                                           "Tanzanite Blue", "Vegas Yellow"])
    test_prompts.extend(color_test)
    
    # Save the prompts
    generator.export_to_csv(test_prompts)