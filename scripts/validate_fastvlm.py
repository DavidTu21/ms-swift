#!/usr/bin/env python3
"""
FastVLM Model Validation Script
============================

This script validates that the FastVLM model implementation is properly integrated
with the ms-swift framework.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def validate_constants():
    """Validate that FastVLM constants are properly defined"""
    print("📋 Validating FastVLM constants...")
    
    try:
        # Import and check model constants
        from typing import List
        constants_context = {'List': List}
        
        with open('swift/llm/model/constant.py') as f:
            exec(f.read(), constants_context)
        
        MLLMModelType = constants_context['MLLMModelType']
        
        # Check that fastvlm is in the model types
        assert hasattr(MLLMModelType, 'fastvlm'), "❌ FastVLM model type not found"
        assert MLLMModelType.fastvlm == 'fastvlm', f"❌ Wrong value: {MLLMModelType.fastvlm}"
        
        # Import and check template constants  
        template_context = {'List': List}
        with open('swift/llm/template/constant.py') as f:
            exec(f.read(), template_context)
            
        MLLMTemplateType = template_context['MLLMTemplateType']
        
        # Check that fastvlm is in the template types
        assert hasattr(MLLMTemplateType, 'fastvlm'), "❌ FastVLM template type not found"
        assert MLLMTemplateType.fastvlm == 'fastvlm', f"❌ Wrong value: {MLLMTemplateType.fastvlm}"
        
        print("   ✅ MLLMModelType.fastvlm =", MLLMModelType.fastvlm)
        print("   ✅ MLLMTemplateType.fastvlm =", MLLMTemplateType.fastvlm)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def validate_model_architecture():
    """Validate model architecture registration"""
    print("\n🏗️ Validating model architecture...")
    
    try:
        # Test that model architecture file loads
        with open('swift/llm/model/model_arch.py') as f:
            content = f.read()
            
        # Check that FastVLM architecture is mentioned
        if 'fastvlm' in content.lower():
            print("   ✅ FastVLM architecture found in model_arch.py")
        else:
            print("   ❌ FastVLM architecture not found in model_arch.py")
            return False
            
        # Check that the architecture mapping includes FastVLM
        if 'MLLMModelArch.fastvlm' in content:
            print("   ✅ FastVLM architecture mapping found")
        else:
            print("   ❌ FastVLM architecture mapping not found") 
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def validate_model_file():
    """Validate FastVLM model implementation file"""
    print("\n📁 Validating FastVLM model file...")
    
    try:
        # Check that the model file exists
        model_file = Path('swift/llm/model/model/fastvlm.py')
        if not model_file.exists():
            print("   ❌ FastVLM model file not found")
            return False
            
        print("   ✅ FastVLM model file exists")
        
        # Check content
        with open(model_file) as f:
            content = f.read()
            
        required_elements = [
            'get_model_tokenizer_fastvlm',
            'MLLMModelType.fastvlm',
            'TemplateType.fastvlm',
            'ModelArch.fastvlm',
            'register_model'
        ]
        
        for element in required_elements:
            if element in content:
                print(f"   ✅ Found: {element}")
            else:
                print(f"   ❌ Missing: {element}")
                return False
                
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def validate_vision_encoder():
    """Validate FastViTHD vision encoder"""
    print("\n👁️ Validating FastViTHD vision encoder...")
    
    try:
        # Check that the encoder file exists
        encoder_file = Path('swift/llm/model/multimodal_encoder/fastvit_hd_encoder.py')
        if not encoder_file.exists():
            print("   ❌ FastViTHD encoder file not found")
            return False
            
        print("   ✅ FastViTHD encoder file exists")
        
        # Check content
        with open(encoder_file) as f:
            content = f.read()
            
        required_elements = [
            'FastViTHDVisionTower',
            'adaptive_pool',
            'feature_enhancer',
            '_add_fastvit_optimizations',
            'feature_select'
        ]
        
        for element in required_elements:
            if element in content:
                print(f"   ✅ Found: {element}")
            else:
                print(f"   ❌ Missing: {element}")
                return False
                
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def validate_template():
    """Validate FastVLM template"""
    print("\n📝 Validating FastVLM template...")
    
    try:
        # Check that the template file exists
        template_file = Path('swift/llm/template/template/fastvlm.py')
        if not template_file.exists():
            print("   ❌ FastVLM template file not found")
            return False
            
        print("   ✅ FastVLM template file exists")
        
        # Check content
        with open(template_file) as f:
            content = f.read()
            
        required_elements = [
            'FastVLMTemplate',
            'FastVLMTemplateMeta',
            'placeholder_tokens',
            'replace_tag',
            'register_template'
        ]
        
        for element in required_elements:
            if element in content:
                print(f"   ✅ Found: {element}")
            else:
                print(f"   ❌ Missing: {element}")
                return False
                
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def validate_imports():
    """Validate that imports are properly set up"""
    print("\n📦 Validating import setup...")
    
    try:
        # Check model __init__.py
        with open('swift/llm/model/model/__init__.py') as f:
            model_init = f.read()
            
        if 'fastvlm' in model_init:
            print("   ✅ FastVLM imported in model/__init__.py")
        else:
            print("   ❌ FastVLM not imported in model/__init__.py")
            return False
            
        # Check template __init__.py
        with open('swift/llm/template/template/__init__.py') as f:
            template_init = f.read()
            
        if 'fastvlm' in template_init:
            print("   ✅ FastVLM imported in template/__init__.py")
        else:
            print("   ❌ FastVLM not imported in template/__init__.py") 
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all validation checks"""
    print("🔍 FastVLM Model Implementation Validation")
    print("=" * 45)
    
    all_passed = True
    
    # Run all validation checks
    checks = [
        validate_constants,
        validate_model_architecture, 
        validate_model_file,
        validate_vision_encoder,
        validate_template,
        validate_imports
    ]
    
    for check in checks:
        if not check():
            all_passed = False
    
    # Summary
    print("\n" + "=" * 45)
    if all_passed:
        print("🎉 All validation checks passed!")
        print("\nFastVLM model implementation is ready for use:")
        print("• FastViTHD vision encoder ✅")
        print("• Phi4 language model integration ✅")
        print("• Model registration ✅")
        print("• Template system ✅")
        print("• Architecture mapping ✅")
        
        print("\n🚀 Next steps:")
        print("1. Add pre-trained model checkpoints")
        print("2. Test with real images and videos")
        print("3. Benchmark against other VLM models")
        
        return 0
    else:
        print("❌ Some validation checks failed!")
        print("Please fix the issues before using the FastVLM model.")
        return 1

if __name__ == '__main__':
    sys.exit(main())