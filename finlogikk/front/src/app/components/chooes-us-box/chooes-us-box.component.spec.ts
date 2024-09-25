import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChooesUsBoxComponent } from './chooes-us-box.component';

describe('ChooesUsBoxComponent', () => {
  let component: ChooesUsBoxComponent;
  let fixture: ComponentFixture<ChooesUsBoxComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ChooesUsBoxComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChooesUsBoxComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
