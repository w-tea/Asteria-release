"""Optional word maps; failure to record never discards a generated image."""
from contextlib import suppress
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from .word_maps import token_contract, MAX_CPU_BYTES, MAX_METADATA
from .word_observer import WordObserver

CONTRACT='asteria.word-maps.v1'


def directory_for(value):
    original=Path(value);directory=original.resolve(strict=True)
    owner_path=directory/'owner.json'
    if original.is_symlink() or owner_path.is_symlink() or owner_path.stat().st_size>MAX_METADATA:
        raise ValueError('word_map_owner_invalid')
    owner=json.loads(owner_path.read_text(encoding='utf-8'))
    root=Path(owner['output_root']).resolve(strict=True)
    if (directory.parent!=root/'_word_maps' or not re.fullmatch(r'[0-9a-f]{32}',directory.name)
            or owner!={'contract':CONTRACT,'output_root':str(root),'operation_id':directory.name}):
        raise ValueError('word_map_directory_invalid')
    return directory


def write_json(directory,name,value):
    data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
    if len(data)>MAX_METADATA: raise ValueError('word_map_metadata_limit')
    temporary=directory/(name+'.tmp');temporary.write_bytes(data);os.replace(temporary,directory/name)


class AsteriaWordCaptureModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('MODEL',),'positive':('CONDITIONING',),'operation':('STRING',)}}
    RETURN_TYPES=('MODEL',)
    FUNCTION='attach'
    CATEGORY='Asteria/internal'

    def attach(self,model,positive,operation):
        import comfy.patcher_extension
        from safetensors.torch import save_file
        config=json.loads(operation)
        directory=directory_for(config['directory'])
        seed=config['seed']
        if type(seed) is not int or not 0<=seed<=2**64-1: raise ValueError('word_map_seed_invalid')
        result=model.clone()
        active=[None]
        try: contract=token_contract(positive,config['profile'])
        except (ValueError,KeyError,TypeError) as exc:
            write_json(directory,f'{seed}.json',{'status':'not_saved','reason':str(exc)})
            return (result,)

        def apply(executor,*args,**kwargs):
            return active[0].apply(executor,*args,**kwargs) if active[0] else executor(*args,**kwargs)

        def outer(executor,noise,latent,sampler,sigmas,*args,**kwargs):
            capture=WordObserver(contract,height=latent.shape[-2],width=latent.shape[-1])
            capture.schedule(sigmas);active[0]=capture
            try:
                output=executor(noise,latent,sampler,sigmas,*args,**kwargs)
                try:
                    tensors,record=capture.result()
                    if tensors is not None:
                        total=sum(p.stat().st_size for p in directory.glob('*.safetensors'))
                        if total+tensors.numel()*tensors.element_size()*2>MAX_CPU_BYTES:
                            raise ValueError('word_map_storage_limit')
                        archive=directory/f'{seed}.safetensors';temporary=archive.with_suffix('.safetensors.tmp')
                        save_file({'maps':tensors.contiguous()},str(temporary))
                        digest=sha256()
                        with temporary.open('rb') as stream:
                            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
                        os.replace(temporary,archive)
                        record.update(archive=archive.name,archive_sha256=digest.hexdigest(),archive_bytes=archive.stat().st_size)
                    record.update(version=1,seed=seed,width=config['width'],height=config['height'])
                    write_json(directory,f'{seed}.json',record)
                except (OSError,ValueError,RuntimeError) as exc:
                    with suppress(OSError,ValueError):
                        write_json(directory,f'{seed}.json',{'status':'not_saved','reason':str(exc)[:256]})
                return output
            finally:
                active[0]=None

        result.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL,'asteria_word_maps',apply)
        result.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,'asteria_word_maps',outer)
        return (result,)


NODE_CLASS_MAPPINGS={'AsteriaWordCaptureModel':AsteriaWordCaptureModel}
