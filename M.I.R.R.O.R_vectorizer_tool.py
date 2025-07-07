# M.I.R.R.O.R Vectorizer Tool - Interactive Concept Vectorization
# Main libraries: sentence-transformers, numpy, scikit-learn
import subprocess
import sys
import json
import os
from datetime import datetime

def install_package(package):
    """Install library if not already installed"""
    try:
        if package == "scikit-learn":
            __import__("sklearn")
        elif package == "sentence-transformers":
            __import__("sentence_transformers")
        else:
            __import__(package.replace('-', '_'))
        print(f"✅ {package} is already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    return True

# Install main libraries
packages_to_install = ["sentence-transformers", "numpy", "scikit-learn", "pandas"]
failed_packages = []

for package in packages_to_install:
    if not install_package(package):
        failed_packages.append(package)

if failed_packages:
    print(f"❌ Failed to install packages: {failed_packages}")
    print("Please install them manually using:")
    for pkg in failed_packages:
        print(f"   pip install {pkg}")
    sys.exit(1)

# Import libraries with error handling
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ All libraries imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all packages are installed:")
    print("   pip install sentence-transformers numpy scikit-learn pandas")
    sys.exit(1)

class MirrorVectorizer:
    def __init__(self, model_name="all-mpnet-base-v2"):
        """Initialize M.I.R.R.O.R Vectorizer with main libraries"""
        print("🚀 Initializing M.I.R.R.O.R Vectorizer...")
        print("📚 Main libraries: sentence-transformers, numpy, scikit-learn")
        self.model = SentenceTransformer(model_name)
        self.vectorized_concepts = []
        self.history_file = "mirror_concepts_history.json"
        self.load_history()
        print("✅ M.I.R.R.O.R Vectorizer is ready!")
    
    def vectorize_concept(self, term, definition, related_terms, alpha=0.4):
        """
        Vectorize M.I.R.R.O.R concept
        
        Args:
            term (str): Concept name
            definition (str): Concept definition
            related_terms (list): List of related terms
            alpha (float): Weight for combination (0-1)
            
        Returns:
            dict: Vectorized concept information
        """
        # Encode related terms
        anchors = [self.model.encode(term, normalize_embeddings=True) for term in related_terms]
        centroid = np.mean(anchors, axis=0)
        
        # Encode definition combined with term
        v_def = self.model.encode(f"{term}: {definition}", normalize_embeddings=True)
        
        # Combine centroid and definition vector
        v0 = (1 - alpha) * centroid + alpha * v_def
        v_hat = v0 / np.linalg.norm(v0)
        
        # Create concept object
        concept = {
            "term": term,
            "definition": definition,
            "related_terms": related_terms,
            "alpha": alpha,
            "vector": v_hat.tolist(),  # Convert numpy array to list for JSON serialization
            "timestamp": datetime.now().isoformat(),
            "vector_norm": float(np.linalg.norm(v_hat)),
            "dimension": len(v_hat)
        }
        
        return concept
    
    def save_concept(self, concept):
        """Save concept to history"""
        self.vectorized_concepts.append(concept)
        self.save_history()
    
    def save_history(self):
        """Save history of vectorized concepts"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.vectorized_concepts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Cannot save history: {e}")
    
    def load_history(self):
        """Load history of vectorized concepts"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.vectorized_concepts = json.load(f)
                print(f"📚 Loaded {len(self.vectorized_concepts)} concepts from history")
        except Exception as e:
            print(f"⚠️ Cannot load history: {e}")
            self.vectorized_concepts = []
    
    def find_similar_concepts(self, new_concept, top_k=3, similarity_threshold=0.75):
        """Find similar concepts"""
        if not self.vectorized_concepts:
            return [], []
        
        new_vector = np.array(new_concept["vector"]).reshape(1, -1)
        similarities = []
        nearest_concepts = []
        
        for concept in self.vectorized_concepts:
            if concept["term"] != new_concept["term"]:  # Exclude itself
                old_vector = np.array(concept["vector"]).reshape(1, -1)
                similarity = cosine_similarity(new_vector, old_vector)[0][0]
                similarities.append((concept, similarity))
                
                # Find "nearest" concepts (cosine > threshold)
                if similarity > similarity_threshold:
                    nearest_concepts.append((concept, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_concepts.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k], nearest_concepts
    
    def display_concept_info(self, concept):
        """Display detailed concept information"""
        print(f"\n📋 CONCEPT INFORMATION:")
        print(f"🔹 Name: {concept['term']}")
        print(f"🔹 Definition: {concept['definition']}")
        print(f"🔹 Related concepts: {', '.join(concept['related_terms'])}")
        print(f"🔹 Alpha: {concept['alpha']}")
        print(f"🔹 Timestamp: {concept['timestamp']}")
        print(f"🔹 Dimension: {concept['dimension']}")
        print(f"🔹 Vector norm: {concept['vector_norm']:.6f}")
        print(f"📊 Vector (first 10 dimensions): {np.array(concept['vector'][:10])}")
    
    def display_similar_concepts(self, similarities):
        """Display similar concepts"""
        if not similarities:
            print("🔍 No similar concepts found in history")
            return
        
        print(f"\n🔍 SIMILAR CONCEPTS:")
        for i, (concept, similarity) in enumerate(similarities, 1):
            print(f"{i}. {concept['term']} (similarity: {similarity:.3f})")
            print(f"   📖 {concept['definition']}")
            print(f"   🔗 {', '.join(concept['related_terms'])}")
    
    def display_nearest_concepts(self, nearest_concepts):
        """Display nearest concepts (cosine > 0.75)"""
        if not nearest_concepts:
            print("🎯 No concepts reached high similarity threshold (>0.75)")
            return
        
        print(f"\n🎯 NEAREST CONCEPTS (cosine > 0.75):")
        print("⚡ These concepts have very high similarity with the new concept!")
        for i, (concept, similarity) in enumerate(nearest_concepts, 1):
            # Icons corresponding to similarity levels
            if similarity > 0.9:
                icon = "🔥"  # Very high
            elif similarity > 0.85:
                icon = "⭐"  # High
            else:
                icon = "✨"  # Fairly high
            
            print(f"{icon} {i}. {concept['term']} (similarity: {similarity:.3f})")
            print(f"   📖 {concept['definition']}")
            print(f"   🔗 {', '.join(concept['related_terms'])}")
            print(f"   📅 {concept['timestamp']}")
            
            # Warning if similarity is too high
            if similarity > 0.95:
                print(f"   ⚠️  WARNING: Very high similarity ({similarity:.3f}) - Possible duplicate!")
            elif similarity > 0.85:
                print(f"   💡 SUGGESTION: Consider relationship with this concept")
            
            print("-" * 50)
    
    def list_all_concepts(self):
        """Display all saved concepts"""
        if not self.vectorized_concepts:
            print("📭 No concepts saved yet")
            return
        
        print(f"\n📚 ALL CONCEPTS LIST ({len(self.vectorized_concepts)} concepts):")
        print("=" * 60)
        
        for i, concept in enumerate(self.vectorized_concepts, 1):
            print(f"{i}. {concept['term']}")
            print(f"   📖 {concept['definition']}")
            print(f"   🔗 {', '.join(concept['related_terms'])}")
            print(f"   📅 {concept['timestamp']}")
            print("-" * 40)
    
    def export_vectors(self, filename="mirror_vectors.csv"):
        """Export vectors to CSV file"""
        if not self.vectorized_concepts:
            print("📭 No concepts to export")
            return
        
        try:
            data = []
            for concept in self.vectorized_concepts:
                row = {
                    'term': concept['term'],
                    'definition': concept['definition'],
                    'related_terms': '|'.join(concept['related_terms']),
                    'timestamp': concept['timestamp'],
                    'alpha': concept['alpha'],
                    'vector_norm': concept['vector_norm']
                }
                # Add each vector dimension
                for i, val in enumerate(concept['vector']):
                    row[f'dim_{i}'] = val
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"📄 Exported {len(data)} concepts to file: {filename}")
        except Exception as e:
            print(f"⚠️ Error exporting file: {e}")
    
    def main_loop(self):
        """Main loop of the tool"""
        print("\n🔄 M.I.R.R.O.R Vectorizer — Interactive Mode")
        print("=" * 60)
        print("📝 Available commands:")
        print("   • Enter concept name to vectorize")
        print("   • 'list' - View all saved concepts")
        print("   • 'export' - Export vectors to CSV file")
        print("   • 'exit' - Exit program")
        print("=" * 60)
        
        while True:
            try:
                command = input("\n📌 Enter command or M.I.R.R.O.R concept name: ").strip()
                
                if command.lower() == 'exit':
                    print("👋 Goodbye! All concepts have been saved.")
                    break
                elif command.lower() == 'list':
                    self.list_all_concepts()
                    continue
                elif command.lower() == 'export':
                    filename = input("📄 Enter filename (Enter for default 'mirror_vectors.csv'): ").strip()
                    if not filename:
                        filename = "mirror_vectors.csv"
                    self.export_vectors(filename)
                    continue
                elif not command:
                    print("⚠️ Please enter a command or concept name")
                    continue
                
                # Process new concept vectorization
                term = command
                definition = input("📖 Enter concept definition: ").strip()
                if not definition:
                    print("⚠️ Definition cannot be empty")
                    continue
                
                related_input = input("🔗 Enter related concepts (separated by commas): ").strip()
                if not related_input:
                    print("⚠️ Must have at least 1 related concept")
                    continue
                
                related_terms = [r.strip() for r in related_input.split(',') if r.strip()]
                
                # Alpha option
                alpha_input = input("⚖️ Enter alpha value (0.1-0.9, Enter for default 0.4): ").strip()
                alpha = 0.4
                if alpha_input:
                    try:
                        alpha = float(alpha_input)
                        if not (0.1 <= alpha <= 0.9):
                            print("⚠️ Alpha must be between 0.1 and 0.9, using default 0.4")
                            alpha = 0.4
                    except ValueError:
                        print("⚠️ Invalid alpha value, using default 0.4")
                        alpha = 0.4
                
                print("\n⏳ Vectorizing...")
                
                # Vectorize concept
                concept = self.vectorize_concept(term, definition, related_terms, alpha)
                
                # Display concept information
                self.display_concept_info(concept)
                
                # Find similar and nearest concepts
                similar, nearest = self.find_similar_concepts(concept)
                
                # Display nearest concepts first (if any)
                self.display_nearest_concepts(nearest)
                
                # Display similar concepts
                self.display_similar_concepts(similar)
                
                # Save concept
                save_choice = input("\n💾 Save this concept? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes', '']:
                    self.save_concept(concept)
                    print("✅ Concept saved successfully!")
                else:
                    print("❌ Concept not saved")
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! All concepts have been saved.")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue

def main():
    """Main function to run the tool"""
    try:
        vectorizer = MirrorVectorizer()
        vectorizer.main_loop()
    except Exception as e:
        print(f"❌ Initialization error: {e}")

if __name__ == "__main__":
    main()